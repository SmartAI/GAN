import os
import tensorflow as tf

from discriminator import Discriminator
from generator import Generator

class DCGAN:

    def __init__(self, sess, FLAGS):
        self.sess = sess
        self.f = FLAGS
        image_shape = [self.f.output_size, self.f.output_size, self.f.c_dim]
        self.real_images = tf.placeholder(tf.float32, [None] + image_shape, name='real_images')
        self.z = tf.placeholder(tf.float32, [None, self.f.z_dim], name='z')

        # initialize models
        generator = Generator(FLAGS)
        discriminator = Discriminator(FLAGS)
        # create network
        self.G = generator(self.z)
        self.D_real, self.D_real_logits = discriminator(self.real_images)
        self.D_fake, self.D_fake_logits = discriminator(self.G, reuse=True)

        # losses
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                self.D_real_logits,
                tf.ones_like(self.D_real))
        )

        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                self.D_fake_logits,
                tf.zeros_like(self.D_fake))
        )

        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                self.D_fake_logits,
                tf.ones_like(self.D_fake))
        )


        self.create_summaries()

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if "d/" in var.name]
        self.g_vars = [var for var in t_vars if "g/" in var.name]

        self.saver = tf.train.Saver()


    def save(self, step):
        model_bname = "DCGAN.model"
        model_dir = self.get_model_dir()
        checkpoint_dir = os.path.join(self.f.checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)


        model_file_prefix = model_dir
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_file_prefix), global_step=step)


    def checkpoint_exists(self):
        model_dir = self.get_model_dir()
        checkpoint_dir = os.path.join(self.f.checkpoint_dir, model_dir)
        print "checkpoint dir is {}".format(checkpoint_dir)
        return os.path.exists(checkpoint_dir)


    def load(self):
        model_dir = self.get_model_dir()
        checkpoint_dir = os.path.join(self.f.checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        return False

    def get_model_dir(self):
        return "{}_{}_{}_{}".format(self.f.experiment_name,
                                    self.f.dataset,
                                    self.f.batch_size,
                                    self.f.output_size)


    def create_summaries(self):
        self.z_sum = tf.summary.histogram("z", self.z)
        self.d_real_sum = tf.summary.histogram("d/output/real", self.D_real)
        self.d_fake_sum = tf.summary.histogram("d/output/fake", self.D_fake)
        self.g_sum = tf.summary.image("generated", self.G, max_outputs=8)
        self.real_sum = tf.summary.image("real", self.real_images, max_outputs=8)
        self.d_loss_real_sum = tf.summary.scalar("d/loss/real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d/loss/fake", self.d_loss_fake)
        self.d_loss_sum = tf.summary.scalar("d/loss/combined", self.d_loss)
        self.g_loss_sum = tf.summary.scalar("g/loss/combined", self.g_loss)
