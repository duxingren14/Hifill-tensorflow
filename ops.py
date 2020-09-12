import logging
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
import glob
import scipy.misc

np.random.seed(2019)
logger = logging.getLogger()

def resize(img, to_shape = None, scale =None, func = None):
    if to_shape is None:
      if scale is None:
        to_shape = img.get_shape().as_list()[1:3]
        to_shape[0], to_shape[1] = to_shape[0] * 2, to_shape[1] * 2
      else:
        to_shape = img.get_shape().as_list()[1:3]
        to_shape[0], to_shape[1] = int(to_shape[0] * scale), int(to_shape[1] * scale)
    return func(img, to_shape)

def scalar_summary(name, scalar):
    tf.summary.scalar(name, scalar)

def images_summary(image, name, max_viz):
    tf.summary.image(name, image[:,:,:,::-1], max_outputs=max_viz)

def gradients_summary(ys, xs, name):
    #print(ys.dtype, xs.dtype)
    grads = tf.gradients(ys, [xs])[0]
    tf.summary.histogram(name, grads)

def flatten(x, name=""):
    return tf.reshape(x, [x.get_shape().as_list()[0], -1], name=name)

def gan_wgan_loss(pos, neg, name):
    d_loss = tf.reduce_mean(neg) - tf.reduce_mean(pos)
    g_loss = -tf.reduce_mean(neg)
    return g_loss, d_loss

def gradients_penalty(interpolates_global, dout_global, mask):
    grad_D_X_hat = tf.gradients(dout_global, [interpolates_global])[0]
    red_idx = np.arange(1, len(interpolates_global.get_shape().as_list())).tolist()
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    return gradient_penalty
 
def random_interpolates(pos, neg):
    epsilon = tf.random_uniform(shape=[pos.get_shape().as_list()[0], 1, 1, 1],
            minval=0.,maxval=1., dtype = tf.float32)
    X_hat = pos + epsilon * (neg - pos)
    return X_hat

def conv2d(x, output_dim, ksize, stride, dilation_rate=1, activation=None, padding='SAME', name='conv', dtype=tf.float32):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [ksize, ksize, x.get_shape().as_list()[-1], output_dim],
                            dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.05))
        conv = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding, \
                       dilations = [1, dilation_rate, dilation_rate, 1])
        biases = tf.get_variable('biases', [output_dim], \
                      dtype=dtype, initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        if activation is None:
            return conv
        else:
            return activation(conv)


def conv2d_ds(x, output_dim, ksize, stride, dilation_rate=1, activation=None, \
                        padding='SAME', name='conv', dtype=tf.float32):
    with tf.variable_scope(name):
      nc = x.get_shape().as_list()[-1]
      #depthwise_filter = tf.get_variable('dw', [3, 3, nc, 1], dtype = dtype, \
      #                 initializer=tf.truncated_normal_initializer(stddev=0.05))
      pointwise_filter = tf.get_variable('pw', [1, 1, nc, output_dim], dtype = dtype, \
                       initializer=tf.truncated_normal_initializer(stddev=0.05))
      #y = tf.nn.separable_conv2d(x, depthwise_filter, pointwise_filter, \
      #            strides = [1, stride, stride, 1], padding = 'SAME', rate=[dilation_rate, dilation_rate])
      y = tf.nn.conv2d(x, pointwise_filter, strides = [1, stride, stride, 1], padding='SAME', \
                 dilations = [1, 1, 1, 1]) 
      biases = tf.get_variable('ds_biases', [output_dim], \
                      dtype=dtype, initializer=tf.constant_initializer(0.0))
      y = tf.reshape(tf.nn.bias_add(y, biases), y.get_shape())
      if activation is None:
            return y
      else:
            return activation(y)


@add_arg_scope
def gen_conv_gated(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', slim=True, activation=None, training=True, dtype=tf.float32):
    x1 = conv2d(x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name+'_feat', dtype=dtype)
    x2 = conv2d(x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name+'_gate', dtype=dtype)
    #x1, x2 = tf.split(x, 2, axis=3) 
    x = tf.sigmoid(x2) * tf.nn.elu(x1)
    return x

@add_arg_scope
def gen_conv_gated_ds(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', slim=True, activation=None, training=True, dtype=tf.float32):
    x1 = conv2d(x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name, dtype=dtype)
    x2 = conv2d_ds(x, cnum, 3, stride, dilation_rate=1,
        activation=None, padding=padding, name=name, dtype=dtype)
    x = tf.sigmoid(x2) * tf.nn.elu(x1)
    return x




@add_arg_scope
def gen_conv_gated_slice(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', slim=True, activation=None, training=True, dtype=tf.float32):
    x1 = conv2d(x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name+'_feat', dtype=dtype)
    x2 = conv2d(x, 1, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name+'_gate', dtype=dtype)
    #x1, x2 = tf.split(x, [cnum,1], axis=3) 
    x = tf.sigmoid(x2) * tf.nn.elu(x1)
    return x

@add_arg_scope
def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.nn.elu, training=True, dtype=tf.float32):
    x = conv2d(x, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name, dtype=dtype)
    return x

@add_arg_scope
def gen_deconv(x, cnum, name='upsample', padding='SAME', training=True, dtype=tf.float32):
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_bilinear)
        x = gen_conv(x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training, dtype=dtype)
    return x

@add_arg_scope
def gen_deconv_gated(x, cnum, name='upsample', padding='SAME', training=True, dtype=tf.float32):
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_bilinear)
        x = gen_conv_gated( x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training, activation=None, dtype=dtype)
    return x

@add_arg_scope
def gen_deconv_gated_ds(x, cnum, name='upsample', padding='SAME', training=True, dtype=tf.float32):
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_bilinear)
        x = gen_conv_gated_ds( x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training, dtype=dtype)
    return x

@add_arg_scope
def gen_deconv_gated_slice(x, cnum, name='upsample', padding='SAME', training=True, dtype=tf.float32):
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_bilinear)
        x = gen_conv_gated_slice(  x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training, dtype=dtype)
    return x

@add_arg_scope
def dis_conv(x, cnum, ksize=5, stride=2, name='conv', training=True, dtype=tf.float32):
    x = conv2d(x, cnum, ksize, stride, padding='SAME', name=name, dtype=dtype)
    x = tf.nn.leaky_relu(x)
    return x

def read_mask_paths(mask_template_dir):
    paths = glob.glob(mask_template_dir+'/*.png')
    return tf.constant(paths, tf.string), len(paths)

def random_rotate_image(image, angle):
    return scipy.misc.imrotate(image, angle, 'nearest')

def random_resize_image(image, scale, height, width):
    newsize = [int(height*scale), int(width*scale)]
    return scipy.misc.imresize(image, newsize, 'nearest')

def filter_gaussian(masks):
    return scipy.ndimage.filters.gaussian_filter(masks[0] ,5)

def random_mask(config, name='mask', dtype=tf.float32):
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img_shape = config.IMG_SHAPE
        height = img_shape[0]
        width = img_shape[1]
        #print('resize', height, width)
        path_list, n_masks = read_mask_paths(config.mask_template_dir)
        nd = tf.random_uniform([ ], minval=0, maxval=n_masks-1, dtype=tf.int32)
        path_mask = path_list[nd]
        contents = tf.read_file(path_mask)
        mask = tf.image.decode_jpeg(contents, channels=3)
        mask = tf.image.random_flip_left_right(mask)
        angle = tf.random_uniform([ ], minval= -90., maxval= 90., dtype=dtype)
        scale = tf.random_uniform([ ], minval=0.8, maxval=1.0, dtype=dtype)
        mask = tf.py_func(random_rotate_image, [mask, angle], tf.uint8)
        mask.set_shape([height, width, 3])
        #print('shape', mask.get_shape().as_list())
        mask = tf.py_func(random_resize_image, [mask, scale, height, width], tf.uint8)
        mask = tf.image.resize_image_with_crop_or_pad(mask, height, width)
        mask = tf.scalar_mul(1./255., tf.cast(tf.expand_dims(mask[:,:,0:1], axis=0), dtype))
        mask.set_shape([1] + [height, width] + [1])
    return mask

def downsample(x, rate):
    shp = x.get_shape().as_list()
    assert shp[1] % rate == 0 and shp[2] % rate == 0, 'height and width should be multiples of rate'
    shp[1], shp[2] = shp[1]//rate, shp[2]//rate
    x = tf.extract_image_patches(x, [1,1,1,1], [1,rate,rate,1], [1,1,1,1], padding='SAME')
    return tf.reshape(x, shp)

def resize_like(mask, x):
    mask_resize = resize(mask, to_shape=x.get_shape().as_list()[1:3],
        func=tf.image.resize_nearest_neighbor)
    return mask_resize

def contextual_attention(src, ref,mask=None,  method='SOFT', ksize=3, rate=1,
                         fuse_k=3, softmax_scale=10., fuse=True, dtype=tf.float32):
    # get shapes
    shape_src = src.get_shape().as_list()
    shape_ref = ref.get_shape().as_list()
    assert shape_src[0] == shape_ref[0] and shape_src[3] == shape_ref[3], 'error'
    batch_size = shape_src[0]
    nc = shape_src[3]

    # raw features
    kernel = rate * 2 - 1
    raw_feats = tf.extract_image_patches(ref, [1,kernel,kernel,1], [1,rate,rate,1], [1,1,1,1], padding='SAME')
    raw_feats = tf.reshape(raw_feats, [batch_size, -1, kernel, kernel, nc])
    raw_feats = tf.transpose(raw_feats, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    raw_feats_lst = tf.split(raw_feats, batch_size, axis=0)

    # resize
    src = downsample(src, rate)
    ref = downsample(ref, rate) 

    ss = tf.shape(src)
    rs = tf.shape(ref)
    shape_s = src.get_shape().as_list()
    shape_r = ref.get_shape().as_list()
    src_lst = tf.split(src, batch_size, axis=0)

    feats = tf.extract_image_patches(ref, [1,ksize,ksize,1], [1,1,1,1], [1,1,1,1], padding='SAME')
    feats = tf.reshape(feats, [batch_size, -1, ksize, ksize, nc])
    feats = tf.transpose(feats, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    feats_lst = tf.split(feats, batch_size, axis=0)

    # process mask
    """
    if mask is None:
        mask = tf.zeros([1]+ shape_ref[1:3] + [1], dtype=dtype)
    mask = resize(mask, to_shape=[32,32], func=tf.image.resize_nearest_neighbor)
    mask = tf.extract_image_patches(mask, [1,ksize,ksize,1], [1,1,1,1], [1,1,1,1], padding='SAME')
    mask = tf.reshape(mask, [1, -1, ksize, ksize, 1])
    mask = tf.transpose(mask, [0, 2, 3, 4, 1])[0]  # bs k k c hw
    mask = tf.cast(tf.equal(tf.reduce_mean(mask, axis=[0,1,2], keepdims=True), 0.), dtype)

    """
    #mask = resize(mask, to_shape=[32,32], func=tf.image.resize_nearest_neighbor)
    mask = tf.nn.max_pool(mask, [1,16,16,1], [1,16,16,1],'SAME')
    mask = tf.nn.max_pool(mask, [1,3,3,1], [1,1,1,1],'SAME')
    mask = 1 - mask
    mask = tf.reshape(mask, [1, 1, 1, -1])


    y_lst, y_up_lst = [], []
    offsets = []
    fuse_weight = tf.reshape(tf.eye(fuse_k, dtype=dtype), [fuse_k, fuse_k, 1, 1])
    for x, r, raw_r in zip(src_lst, feats_lst, raw_feats_lst):
        r = r[0]
        r = r / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(r), axis=[0,1,2])), 1e-8)
        y = tf.nn.conv2d(x, r, strides=[1,1,1,1], padding="SAME")

        if fuse:
            yi = tf.reshape(y, [1, ss[1]*ss[2], rs[1]*rs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, ss[1], ss[2], rs[1], rs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, ss[1]*ss[2], rs[1]*rs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, ss[2], ss[1], rs[2], rs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            y = yi
        y = tf.reshape(y, [1, ss[1], ss[2], rs[1]*rs[2]])
        if method == 'HARD':
            ym = tf.reduce_max(y, keepdims=True, axis=3) 
            y = y * mask
            coef = tf.cast( tf.greater_equal(y , tf.reduce_max(y, keepdims=True, axis=3)), dtype) 
            y =  tf.pow( coef * tf.divide(y, ym + 1e-04 ), 2)
        elif method == 'SOFT':
            y = tf.nn.softmax(y * mask * softmax_scale, 3) * mask
        y.set_shape([1, shape_s[1], shape_s[2], shape_r[1]*shape_r[2]])

        if dtype == tf.float32:
            offset = tf.argmax(y, axis=3, output_type=tf.int32)
            offsets.append(offset)
        feats = raw_r[0]
        y_up = tf.nn.conv2d_transpose(y, feats, [1] + shape_src[1:], strides=[1,rate,rate,1]) 
        y_lst.append(y)
        y_up_lst.append(y_up)

    out, correspondence = tf.concat(y_up_lst, axis=0), tf.concat(y_lst, axis=0)
    out.set_shape(shape_src)

    #print(correspondence.get_shape().as_list())
    #correspondence.reshape([ss[0], ss[1], ss[2], -1])
    if dtype == tf.float32:
        offsets = tf.concat(offsets, axis=0)
        offsets = tf.stack([offsets // ss[2], offsets % ss[2]], axis=-1)
        offsets.set_shape(shape_s[:3] + [2])
        h_add = tf.tile(tf.reshape(tf.range(ss[1]), [1, ss[1], 1, 1]), [ss[0], 1, ss[2], 1])
        w_add = tf.tile(tf.reshape(tf.range(ss[2]), [1, 1, ss[2], 1]), [ss[0], ss[1], 1, 1])
        offsets = offsets - tf.concat([h_add, w_add], axis=3)
        flow = flow_to_image_tf(offsets)
        flow = resize(flow, scale=rate, func=tf.image.resize_nearest_neighbor)
    else:
        flow = None
    return out, correspondence, flow

def apply_contextual_attention(x, mask_s, method = 'SOFT', name='attention', dtype=tf.float32, conv_func = None):
    x_hallu = x
    sz = x.get_shape().as_list()[1]
    nc = x.get_shape().as_list()[3]
    x, corres, flow = contextual_attention(x, x, mask_s, method = method, ksize=3, rate=2, fuse=True, dtype=dtype)
    x = conv_func(x, nc, 3, 1, name= name + '_att1')
    #x = conv_func(x, nc, 3, 1, name= name + '_att2')
    x = tf.concat([x_hallu, x], axis=3)
    x = conv_func(x, nc, 3, 1, name= name + '_att3')
    #x = conv_func(x, nc, 3, 1, name= name + '_att4')
    return x, corres, flow

def apply_attention(x, correspondence, conv_func, name):
    shp = x.get_shape().as_list()
    shp_att = correspondence.get_shape().as_list()
    #print(shp, shp_att)
    rate = shp[1]// shp_att[1]
    kernel = rate * 2
    batch_size = shp[0]
    sz = shp[1]
    nc = shp[3]
    raw_feats = tf.extract_image_patches(x, [1,kernel,kernel,1], [1,rate,rate,1], [1,1,1,1], padding='SAME')
    raw_feats = tf.reshape(raw_feats, [batch_size, -1, kernel, kernel, nc])
    raw_feats = tf.transpose(raw_feats, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    raw_feats_lst = tf.split(raw_feats, batch_size, axis=0)
    
    ys = []
    att_lst = tf.split(correspondence, batch_size, axis=0)
    for feats, att in zip(raw_feats_lst, att_lst):
        #print(att.get_shape().as_list(), feats.get_shape().as_list())
        y = tf.nn.conv2d_transpose(att, feats[0], [1] + shp[1:], strides=[1,rate,rate,1])
        ys.append(y)
    out = tf.concat(ys, axis=0)
    if conv_func is not None:
      out = conv_func(out, nc, 3, 1, rate=1, name = name + '_1')
      out = conv_func(out, nc, 3, 1, rate=2, name = name + '_2')
    return out


def apply_attention2(x, correspondence, name):
    shp = x.get_shape().as_list()
    shp_att = correspondence.get_shape().as_list()
    #print(shp, shp_att)
    rate = shp[1]// shp_att[1]
    kernel = rate
    batch_size = shp[0]
    sz = shp[1]
    nc = shp[3]
    raw_feats = tf.extract_image_patches(x, [1,kernel,kernel,1], [1,rate,rate,1], [1,1,1,1], padding='SAME')
    raw_feats = tf.reshape(raw_feats, [batch_size, -1, kernel, kernel, nc])
    raw_feats = tf.transpose(raw_feats, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    raw_feats_lst = tf.split(raw_feats, batch_size, axis=0)
    
    ys = []
    att_lst = tf.split(correspondence, batch_size, axis=0)
    for feats, att in zip(raw_feats_lst, att_lst):
        #print(att.get_shape().as_list(), feats.get_shape().as_list())
        y = tf.nn.conv2d_transpose(att, feats[0], [1] + shp[1:], strides=[1,rate,rate,1])
        ys.append(y)
    out = tf.concat(ys, axis=0)
    return out

def residual_block(x, name, conv_func):
    sz = x.get_shape().as_list()[1]
    nc = x.get_shape().as_list()[3]
    x1 = conv_func(x, nc, 3, 1, name= name + '_res')
    return x + x1


def dilate_block(x, name, conv_func):
    sz = x.get_shape().as_list()[1]
    nc = x.get_shape().as_list()[3]
    x = conv_func(x, nc, 3, 1, name= name + '_d1')
    x = conv_func(x, nc, 3, rate=1, name= name + '_d2')
    x = conv_func(x, nc, 3, rate=1, name= name + '_d3')
    x = conv_func(x, nc, 3, rate=2, name= name + '_d4')
    x = conv_func(x, nc, 3, rate=2, name= name+ '_d5')
    x = conv_func(x, nc, 3, rate=2, name= name + '_d6')
    x = conv_func(x, nc, 3, rate=2, name= name + '_d7')
    x = conv_func(x, nc, 3, rate=2, name= name + '_d8')
    x = conv_func(x, nc, 3, rate=4, name= name + '_d9')
    x = conv_func(x, nc, 3, rate=4, name= name + '_d10')
    x = conv_func(x, nc, 3, rate=4, name= name + '_d11')
    x = conv_func(x, nc, 3, rate=4, name= name+ '_d12')
    x = conv_func(x, nc, 3, rate=8, name= name + '_d13')
    x = conv_func(x, nc, 3, rate=8, name= name + '_d14')
    return x
"""
def dilate_block(x, name, conv_func):
    sz = x.get_shape().as_list()[1]
    nc = x.get_shape().as_list()[3]
    x = conv_func(x, nc, 3, 1, name= name + '_d1')
    x = conv_func(x, nc, 3, rate=2, name= name + '_d2')
    x = conv_func(x, nc, 3, rate=4, name= name+ '_d4')
    x = conv_func(x, nc, 3, rate=8, name= name + '_d8')
    x = conv_func(x, nc, 3, rate=16, name= name + '_d16')
    x = conv_func(x, nc, 3, rate=16, name= name + '_d16_2')
    x = conv_func(x, nc, 3, rate=8, name= name+ '_d8_2')
    x = conv_func(x, nc, 3, rate=4, name= name + '_d4_2')
    x = conv_func(x, nc, 3, rate=2, name= name + '_d2_2')
    x = conv_func(x, nc, 3, rate=1, name= name + '_d1_2')
    return x

"""

def dilate_block2(x, name, conv_func):
    sz = x.get_shape().as_list()[1]
    nc = x.get_shape().as_list()[3]
    #conv_func = gen_conv_gated
    x = conv_func(x, nc, 3, 1, name= name + '_d1')
    x = conv_func(x, nc, 3, rate=2, name= name + '_d2')
    x = conv_func(x, nc, 3, rate=4, name= name+ '_d4')
    x = conv_func(x, nc, 3, rate=8, name= name + '_d8')
    x = conv_func(x, nc, 3, rate=16, name= name + '_d16')
    return x


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def compute_color(u,v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img

def flow_to_image(flow):
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))

def flow_to_image_tf(flow, name='flow_to_image'):
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(flow_to_image, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 127.5 - 1.
        return img

