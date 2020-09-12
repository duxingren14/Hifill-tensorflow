
import tensorflow as tf
import time
import os
from model import HinpaintModel
from tensorflow.python.framework import ops
from time import sleep
import numpy as np
		
def read_images(input_queue):
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    example = tf.reverse(example, [2])
    return example

def preprocess_image(image, sz):
    return tf.image.resize_images(image, sz)

def get_input_queue(fnames):
  input_queue = tf.RandomShuffleQueue(capacity=100000, min_after_dequeue=16, dtypes = [tf.string])
  enqueue_op = input_queue.enqueue_many([  [[fn] for fn in fnames]])
  #tf.train.slice_input_producer([fnames], shuffle=True)
  return input_queue, enqueue_op


def get_batch(input_queue, config):
  #input_queue = tf.train.slice_input_producer([fname_ph],
  #                                          shuffle=True)
  #input_queue = tf.RandomShuffleQueue(capacity=128, min_after_dequeue=int(0.9*128), dtypes = [tf.string])
  #enqueue_op = input_queue.enqueue_many([fname_ph])
  fn = input_queue.dequeue()
  image = read_images(fn)
  image = preprocess_image(image, config.IMG_SHAPE[:-1])
  image_batch = tf.train.batch([image], num_threads=2,
    capacity=32, batch_size=config.BATCH_SIZE, allow_smaller_final_batch=False)
  return image_batch

def d_graph_deploy(model, data, config, gpu_id=0):
    #images  = get_batch(input_queue, config)
    images = data.data_pipeline(config.BATCH_SIZE)
    _, _, losses = model.build_graph_with_losses(images, config, reuse=True)
    return losses['d_loss']

def g_graph_deploy(model, data, config, gpu_id=0):
    #images = get_batch(input_queue, config)
    images = data.data_pipeline(config.BATCH_SIZE)
    if gpu_id == 0:
        _, _, losses = model.build_graph_with_losses(
            images, config, summary=True, reuse=True)
    else:
        _, _, losses = model.build_graph_with_losses(
            images, config, reuse=True)
    return losses['g_loss']

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def def_multigpu_trainer(gpu_ids, optimizer, graph_def, graph_def_kwds, var_list):
    losses = []
    tower_grads = []
    for gpu_id in gpu_ids:
        with tf.device('/device:GPU:'+str(gpu_id)):
            l = graph_def(** graph_def_kwds, gpu_id = gpu_id)
            grads = optimizer.compute_gradients(l, var_list = var_list)
            tower_grads.append(grads)
            losses.append( l )
    loss = tf.add_n(losses)
    tower_grads = average_gradients(tower_grads)
    train_op = optimizer.apply_gradients(tower_grads)
    return loss, train_op


class HiTrainer:
    def __init__(self,
        gen_kwargs,
        dis_kwargs,
        saver_kwargs,
        summary_kwargs,
        enq_ops
    ):
        self.gen_kwargs = gen_kwargs
        self.dis_kwargs = dis_kwargs
        self.saver_kwargs = saver_kwargs
        self.summary_kwargs = summary_kwargs

        conf = tf.ConfigProto(log_device_placement=False, allow_soft_placement = True)
        conf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=conf)
        self.summary_writer = tf.summary.FileWriter(self.summary_kwargs['dir'], self.sess.graph)

        self.def_d_optimizer()
        self.def_g_optimizer()
        self.enq_ops = enq_ops

        all_var_list = self.gen_kwargs['var_list'] + self.dis_kwargs['var_list']
        self.saver = tf.train.Saver(all_var_list)
        self.summaries = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        if self.saver_kwargs['restore_dir'] is not None:
            ckpt = tf.train.get_checkpoint_state(self.saver_kwargs['restore_dir'])
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.saver_kwargs['restore_dir'], ckpt_name))
        
        
    def def_d_optimizer(self):
        if self.dis_kwargs['gpu_ids'] is None:
            gpu_ids = [0]
        else:
            gpu_ids = self.dis_kwargs['gpu_ids']
        self.d_loss, self.d_train_op = def_multigpu_trainer(
                         gpu_ids, 
                         self.dis_kwargs['optimizer'], 
                         self.dis_kwargs['graph_def'], 
                         self.dis_kwargs['graph_def_kwargs'], 
                         self.dis_kwargs['var_list'])

    def def_g_optimizer(self):
        if self.gen_kwargs['gpu_ids'] is None:
            gpu_ids = [0]
        else:
            gpu_ids = self.gen_kwargs['gpu_ids']
        self.g_loss, self.g_train_op = def_multigpu_trainer(
                         gpu_ids, 
                         self.gen_kwargs['optimizer'], 
                         self.gen_kwargs['graph_def'], 
                         self.gen_kwargs['graph_def_kwargs'], 
                         self.gen_kwargs['var_list'])

    def run_summary_writer(self, step):
        if step % self.summary_kwargs['period'] == 0:
            sum_str = self.sess.run( self.summaries)
            self.summary_writer.add_summary(sum_str, global_step = step)
        
    def run_saver(self,step):
        if step % self.saver_kwargs['period'] == 0:
            self.saver.save(self.sess, self.saver_kwargs['dir'], global_step = step)

    def run_d_optimizer(self):
        for i in range(self.dis_kwargs['max_iters']):
            __, dloss = self.sess.run([self.d_train_op, self.d_loss])
        return dloss

    def run_g_optimizer(self):
        __, gloss = self.sess.run([self.g_train_op, self.g_loss])
        return gloss
    
    def train(self):
        start_time = time.time()
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(self.sess, coord=coord)
        for step in range(self.gen_kwargs['max_iters']):
            dloss = self.run_d_optimizer()
            gloss = self.run_g_optimizer()
            self.run_summary_writer(step)
            self.run_saver(step)
            if step % 10 == 0:
                tpb = (time.time() - start_time) /(step + 1)
                print("Epoch %d, Step %d, tpb: %.8lf, dloss:%.4lf, gloss:%.4lf" \
                    %(step//self.gen_kwargs['SPE'], step, tpb, dloss, gloss))
        print('total time spent: ', time.time() - start_time)
        coord.request_stop()
        coord.join(threads)
        self.sess.close() 
        
