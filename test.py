import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import scipy.misc
import glob 
import os
import random
import time


from scipy import signal

import progressbar
from time import sleep
from model import HinpaintModel
from ops import resize, apply_attention2
from utils import load_yml

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default='', type=str,
                    help='The directory of images to be completed.')
parser.add_argument('--mask_dir', default='', type=str,
                    help='The directory of masks, value 255 indicates mask.')
parser.add_argument('--output_dir', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')
parser.add_argument('--rectangle_mask', default=True, type=bool,
                    help='whether to use rectangle masks.')
parser.add_argument('--input_size', default=512, type=int,
                    help='The size of input image.')
parser.add_argument('--times', default=1, type=int,
                    help='The size of input image.')


args = parser.parse_args()


def sort(str_lst):
    return [s for s in sorted(str_lst)]

def read_imgs_masks(args):
    paths_img = glob.glob(args.image_dir+'/*.*[g|G]')
    paths_img = sort(paths_img)
    paths_mask = glob.glob(args.mask_dir+'/*.*[g|G]')
    paths_mask = sort(paths_mask)
    return paths_img, paths_mask
    
def get_input(path_img, path_mask):
    #image = cv2.resize(cv2.imread(path_img), (args.input_size * args.times, args.input_size * args.times))
    #mask = cv2.resize(cv2.imread(path_mask), (args.input_size * args.times, args.input_size * args.times), interpolation=cv2.INTER_NEAREST)
    #mask = 255 - mask
    image = cv2.imread(path_img)
    mask = cv2.imread(path_mask)
   
    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    return np.concatenate([image, mask], axis=2), image[0], mask[0]

dtype = tf.float32
def build_inference_net(model, args):
        raw_img_ph = tf.placeholder(shape=None, dtype=dtype)
        raw_mask_ph = tf.placeholder(shape=None, dtype=dtype)
        raw_img = tf.expand_dims(raw_img_ph, axis=0)
        #print(raw_img.dtype, 'raw image')
        large_img = resize(raw_img, to_shape=[args.times * args.input_size, args.times * args.input_size], func=tf.image.resize_nearest_neighbor)
        large_img.set_shape([1, args.times * args.input_size, args.times * args.input_size, 3])
        large_img = large_img/127.5 - 1
        small_img =  tf.extract_image_patches(
                         large_img, 
                         [1,args.times , args.times ,1], 
                         [1, args.times,args.times,1], 
                         [1,1,1,1], 
                         padding='SAME')
        small_img = tf.reshape(small_img, [1, args.input_size, args.input_size, args.times, args.times, 3])
        small_img = tf.reduce_mean(small_img, axis=[3,4])

        raw_mask = tf.expand_dims(raw_mask_ph, axis=0)
        small_mask = resize(raw_mask, to_shape=[args.input_size, args.input_size], func=tf.image.resize_nearest_neighbor)
        small_mask.set_shape([1, args.input_size, args.input_size, 3])
        small_mask = 1 - small_mask/255
        x2, x2r, corres = model.build_inference_graph(small_img, small_mask, dtype=dtype, config=config)

        small_output = (x2 + 1.) * 127.5
        small_output = tf.saturate_cast(small_output, tf.uint8)
        large_output, out1, out2, out3= post_processing(large_img, small_img, x2, small_mask, corres, args)
        raw_size_output = resize_back(raw_img, large_output, small_mask)
        return raw_size_output, raw_img_ph, raw_mask_ph




def gaussian_kernel(size, std):
    k = signal.gaussian(size, std)
    kk = np.matmul(k[:, np.newaxis], [k])
    #print(kk.shape)
    return kk/np.sum(kk)
    

def resize_back(raw_img, large_output, small_mask):
    raw_shp = tf.shape(raw_img)
    raw_size_output = resize(large_output, to_shape=raw_shp[1:3], func=tf.image.resize_bilinear)
    raw_size_output = tf.cast(raw_size_output, dtype)

    gauss_kernel = gaussian_kernel(7,  1.)
    gauss_kernel = tf.cast(gauss_kernel, dtype)
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    mask = tf.nn.conv2d(small_mask[:,:,:,0:1], gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
    mask = resize(mask,  to_shape=[raw_shp[1], raw_shp[2]], func=tf.image.resize_bilinear)
    mask = tf.cast(mask, dtype)

    raw_size_output = raw_size_output * mask + raw_img * (1-mask)
    raw_size_output = tf.saturate_cast(raw_size_output, tf.uint8)
    return raw_size_output


def post_processing(large_img, small_img, low_base, small_mask, corres, args):

    high_raw = large_img
    low_raw = small_img
    mask = 1 - small_mask
    low_raw = resize(low_raw, scale=args.times, func=tf.image.resize_bilinear)
    mask = resize(mask, scale=args.times, func=tf.image.resize_nearest_neighbor)
    residual1 = (high_raw - low_raw) * mask
    residual = apply_attention2(residual1, corres, 'generator/lp') 

    low_base = resize(low_base, scale=args.times, func=tf.image.resize_bilinear)
    x = low_base + residual
    x = tf.clip_by_value(x, -1., 1.)
    x = (x + 1.) * 127.5
    return x, low_raw, low_base, residual



if __name__ == "__main__":
    config = load_yml('config.yml')
    paths_img, paths_mask = read_imgs_masks(args)
    #ng.get_gpus(0)
    model = HinpaintModel()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with tf.Session(config=sess_config) as sess:
        outputs, raw_img_ph, raw_mask_ph = build_inference_net(model, args)

        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))

        sess.run(assign_ops)
        print('Model loaded.')
        total_time = 0.

        bar = progressbar.ProgressBar(maxval=len(paths_img), \
             widgets=[progressbar.Bar('=', '[', ']'), ' ', \
             progressbar.Percentage()])
        bar.start()
        for (i, path_img) in enumerate(paths_img):
            rint = i % len(paths_mask)
            bar.update(i+1)
            in_img, img, mask = get_input(path_img, paths_mask[rint])
            s = time.time()

            outputs_arr= sess.run(outputs, feed_dict={raw_img_ph : img, raw_mask_ph : 255 - mask})
            res = outputs_arr[0]
            total_time += time.time() - s
            img_hole = img * (1-mask/255) + mask 
            res = np.concatenate([img, img_hole, res], axis=1)
            cv2.imwrite(args.output_dir + '/' + str(i)+ '.jpg', res)
        bar.finish()
        print('average time per image', total_time/len(paths_img))
