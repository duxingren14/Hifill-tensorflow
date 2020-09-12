import cv2
import scipy.ndimage
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from ops import scalar_summary, images_summary
from ops import gradients_summary
from ops import flatten, resize
from ops import gan_wgan_loss, gradients_penalty
from ops import random_interpolates

from ops import gen_conv_gated, gen_deconv_gated, dis_conv, gen_deconv, gen_conv
from ops import gen_deconv_gated_slice, gen_conv_gated_slice, gen_conv_gated_ds, gen_deconv_gated_ds 
from ops import random_mask
from ops import resize_like, contextual_attention
from ops import apply_attention, dilate_block, residual_block, apply_contextual_attention
from ops import filter_gaussian, dilate_block2


def get_conv_op(conv_type):
    #print(conv_type, 'ds')
    if conv_type == 'none':
        conv = gen_conv
        deconv = gen_deconv
    elif conv_type == 'regular':
        conv = gen_conv_gated
        deconv = gen_deconv_gated
    elif conv_type == 'ds':
        conv = gen_conv_gated_ds
        deconv = gen_deconv_gated_ds
    elif conv_type == 'slice':
        conv = gen_conv_gated_slice
        deconv = gen_deconv_gated_slice
    else:
        raise('wrong conv type ' + conv_type)
    return conv, deconv

class HinpaintModel:
    def __init__(self):
        self.name = 'Hinpaint'
        #super().__init__('HinPaint')

    def build_generator(self, x, mask, config=None, reuse=False,
                          training=True, padding='SAME', name='generator', dtype=tf.float32):
        
        x_in = x
        mask_batch = tf.ones(x_in.get_shape().as_list()[0:3]+[1], dtype=dtype) * mask
        x = tf.concat([x_in, mask_batch], axis=3)
        
        # conv and deconv for stage-1
        conv1, deconv1 = get_conv_op(config.COARSE_CONV_TYPE)

        # conv and deconv for stage-2
        conv2, deconv2 = get_conv_op(config.REFINE_CONV_TYPE)

        # two-stage
        sz = config.IMG_SHAPE[1]
        nc = config.GEN_NC 
        offset_flow = None
        with tf.variable_scope(name, reuse=reuse), \
                arg_scope([conv1, deconv1, conv2, deconv2], training=training, padding=padding, dtype=dtype):
            # stage-1
            x = resize(x, to_shape=[256, 256], func=tf.image.resize_bilinear)
            x = conv1(x, nc, 5, 2, name='c_en_down_128')
            x = conv1(x, nc, 3, 1, name='c_en_conv_128')
            x = conv1(x, 2*nc, 3, 2, name='c_en_down_64')
            x = conv1(x, 2*nc, 3, 1, name='c_en_conv1_64')
            x = conv1(x, 2*nc, 3, 1, name='c_en_conv2_64')
            x = conv1(x, 2*nc, 3, 1, name='c_en_conv3_64')
            x = dilate_block(x, name='c_dil', conv_func = conv1)
            x = conv1(x, 2*nc, 3, 1, name='c_de_conv1_64')
            x = conv1(x, 2*nc, 3, 1, name='c_de_conv2_64')
            x = conv1(x, 2*nc, 3, 1, name='c_de_conv3_64')
            x = deconv1(x, nc, name='c_de_up_128')
            x = conv1(x, nc, 3, 1, name='c_de_conv_128')
            x = deconv1(x, 3, name='c_de_toRGB')
            x = tf.clip_by_value(x, -1., 1.)

            x = resize(x, to_shape=x_in.get_shape().as_list()[1:3], func=tf.image.resize_bilinear)
            #x = tf.cast(x, dtype)
            x.set_shape(x_in.get_shape().as_list())
            x1 = x
            x_coarse = x * mask_batch + x_in * (1.-mask_batch)

            # stage-2
            xnow = tf.concat([x_coarse, mask_batch], axis=3)
            activations = [x_coarse]
            # encoder
            sz_t = sz
            x = xnow
            nc = max(4, nc//(sz//512)) //2
            while sz_t > config.BOTTLENECK_SIZE:
                nc *= 2
                sz_t //= 2
                kkernal = 5 if sz_t == sz else 3
                x = conv2(x, nc, 3, 2, name='re_en_down_' + str(sz_t))
                x = conv2(x, nc, 3, 1, rate=1, name='re_en_conv_'+str(sz_t))
                activations.append(x)

            # dilated conv
            x = dilate_block2(x, name = 're_dil', conv_func = conv2)

            # attention 
            mask_s = mask #resize_like(mask, x)
            x, match, offset_flow = apply_contextual_attention(x, mask_s, method = config.ATTENTION_TYPE, \
                             name='re_att_'+str(sz_t), dtype=dtype, conv_func=conv2)
            # decoder
            activations.pop(-1)
            while sz_t < sz//2:
                nc = nc//2 
                sz_t *= 2
                x = deconv2(x, nc, name='re_de_up__'+str(sz_t))
                x = conv2(x, nc, 3, 1, rate=1, name='re_de_conv_'+str(sz_t))
                x_att = apply_attention(activations.pop(-1), match, conv_func = conv2, name='re_de_att_' + str(sz_t))
                x = tf.concat([x_att, x], axis=3)
            x = deconv2(x, 3, name='re_de_toRGB__'+str(sz_t))
            x2 = tf.clip_by_value(x, -1., 1.)
        if training:
            return  x1, x2, offset_flow 
        else:
            return x1, x2, match, offset_flow


    def build_discriminator(self, x, reuse=False, training=True, nc=64):
        with tf.variable_scope('discriminator', reuse=reuse):
            x = dis_conv(x, nc, name='conv1', training=training)
            x = dis_conv(x, nc*2, name='conv2', training=training)
            x = dis_conv(x, nc*4, name='conv3', training=training)
            x = dis_conv(x, nc*4, name='conv4', training=training)
            x = dis_conv(x, nc*4, name='conv5', training=training)
            x = dis_conv(x, nc*4, name='conv6', training=training)
            x = flatten(x, name='reshape')
            D = tf.layers.dense(x, 1, name='linear')
            return D

    def build_graph_with_losses(self, real, config, training=True, summary=False, reuse=False):
        real = real / 127.5 - 1.
        mask = random_mask(config, name='mask_input')
        x = real * (1.-mask)
        x1, x2, offset_flow = self.build_generator(
            x, mask, config, reuse=reuse, training=training)
        fake = x2
        losses = {}
        # apply mask and reconstruct
        fake_patched = fake * mask + x * (1.-mask)

        coarse_alpha = config.COARSE_ALPHA
        losses['l1_loss'] = coarse_alpha * tf.reduce_mean(tf.abs(real - x1)* mask)
        losses['l1_loss'] += tf.reduce_mean(tf.abs(real - x2)* mask)

        losses['ae_loss'] = coarse_alpha * tf.reduce_mean(tf.abs(real - x1) * (1.-mask))
        losses['ae_loss'] += tf.reduce_mean(tf.abs(real - x2)* (1.-mask) )
        losses['ae_loss'] /= tf.reduce_mean(1.-mask)
        if summary:
            viz_img = [real, x, x1, x2, fake_patched]
            if offset_flow is not None:
                viz_img.append(resize(offset_flow, to_shape=config.IMG_SHAPE[0:2], func=tf.image.resize_nearest_neighbor))
            images_summary(tf.concat(viz_img, axis=2), 'train_real_x_x1_x2_result_flow', config.VIZ_MAX_OUT)

        # gan
        real_fake = tf.concat([real, fake_patched], axis=0)
        if config.GAN_WITH_MASK:
            real_fake = tf.concat([real_fake, tf.tile(mask, [config.BATCH_SIZE*2, 1, 1, 1])], axis=3)

        # gan loss
        D_real_fake = self.build_discriminator(real_fake, training=training, reuse=reuse, nc=config.DIS_NC)
        D_real, D_fake = tf.split(D_real_fake, 2)
        g_loss, d_loss = gan_wgan_loss(D_real, D_fake, name='gan_loss')
        losses['g_loss'] = g_loss
        losses['d_loss'] = d_loss
        # gp
        interps = random_interpolates(real, fake_patched)
        D_interps = self.build_discriminator(interps, reuse=True, nc=config.DIS_NC)
        # apply gp
        gp_loss = gradients_penalty(interps, D_interps, mask=mask)
        losses['gp_loss'] = config.WGAN_GP_LAMBDA * gp_loss
        losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
        if summary:
          gradients_summary(g_loss, fake, name='g_loss_to_fake')
          scalar_summary('d_loss_with_gp', losses['d_loss'])
          scalar_summary('d_loss', d_loss)
          scalar_summary('g_loss', g_loss)
          scalar_summary('d_loss', d_loss)
          scalar_summary('l1', losses['l1_loss'])
          scalar_summary('ae', losses['ae_loss'])
        losses['g_loss'] = config.GAN_LOSS_ALPHA * losses['g_loss']
        losses['g_loss'] += config.L1_LOSS_ALPHA * losses['l1_loss']
        losses['g_loss'] += config.AE_LOSS_ALPHA * losses['ae_loss']
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

    def build_static_graph(self, real, config, mask=None, name='val'):
        mask = random_mask(config, name=name+'mask_input')
        real = real / 127.5 - 1.
        edges = None
        x = real*(1.-mask)
        # inpaint
        x1, x2, __, offset_flow = self.build_generator(
            x, mask, config, reuse=True, training=False)
        fake = x2
        # apply mask and reconstruct
        fake_patched = fake * mask + x*(1.-mask)
        # image visualization
        viz_img = [real, x, x1, x2, fake_patched]
        if offset_flow is not None:
            viz_img.append(resize(offset_flow,  to_shape=config.IMG_SHAPE[0:2],
                       func=tf.image.resize_nearest_neighbor))
        images_summary(tf.concat(viz_img, axis=2),
            name+'_real_x_x1_x2_result_flow', config.VIZ_MAX_OUT)
        return fake_patched

    def build_inference_graph(self, real, mask, config=None, reuse=False, is_training=False, dtype=tf.float32):
        mask = mask[0:1, :, :, 0:1]
        x = real * (1. - mask)
        x1, x2, corres, flow = self.build_generator(x, mask, config=config, reuse=reuse, training=is_training,
            dtype=dtype)
        fake = x2
        fake_patched = fake * mask + x * (1-mask)
        return x2, fake_patched, corres
