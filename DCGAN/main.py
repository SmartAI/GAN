import os
import pprint
pp = pprint.PrettyPrinter()

import tensorflow as tf
from dcgan import DCGAN
from train import train
import inference


flags = tf.app.flags

# trainning params
flags.DEFINE_integer('epoch', 25, "Number of epochs to train")
flags.DEFINE_float('learning_rate', 0.0002, "Learning rate for Adam optimizer")
flags.DEFINE_float('beta1', 0.5, "Momentum term of Adam optimizer")
flags.DEFINE_integer('batch_size', 64, "Number of images in batch")


# model params
flags.DEFINE_integer("output_size", 64, "size of output images to produce")
flags.DEFINE_integer("z_dim", 100, "dimension of input noise vector")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color")
flags.DEFINE_integer('gf_dim', 64, "Dimension of generator filters in first conv layer")
flags.DEFINE_integer('df_dim', 64, "Dimension of discriminator filters in first conv layer")

# dataset params
flags.DEFINE_string("data_dir", "data", "Path to datasets directory")
flags.DEFINE_string("dataset", "faces", "The name of dataset")

# flags for running
flags.DEFINE_string("experiment_name", "experiment", "Name of experiment for current run")
flags.DEFINE_boolean("train", False, "Train if True, otherwise test")
flags.DEFINE_integer("sample_size", 64, "Number of image to sample")

# directory params
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Path to save the checkpoint data")
flags.DEFINE_string("sample_dir", "samples", "Path to save the image samples")
flags.DEFINE_string("log_dir", "logs", "Path to log for Tensorboard")
flags.DEFINE_string("image_ext", "jpg", "image extension to find")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(FLAGS.__flags)
    with tf.Session() as sess:
        dcgan = DCGAN(sess, FLAGS)

        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)
        if not os.path.exists(os.path.join(FLAGS.sample_dir, dcgan.get_model_dir())):
            os.makedirs(os.path.join(FLAGS.sample_dir, dcgan.get_model_dir()))
        if not os.path.exists(os.path.join(FLAGS.log_dir, dcgan.get_model_dir())):
            os.makedirs(os.path.join(FLAGS.log_dir, dcgan.get_model_dir()))

        if dcgan.checkpoint_exists():
            print "Loading checkpoints"
            if dcgan.load():
                print "Success"
            else:
                raise IOError("Could not read checkpoints from {}".format(
                    FLAGS.checkpoint_dir))
        else:
            if not FLAGS.train:
                raise IOError("No checkpoints found")
            print "No checkpoints found. Training from scratch"
            dcgan.load()

        if FLAGS.train:
            train(dcgan)

        print "Generating samples..."
        inference.sample_images(dcgan)
        inference.visualize_z(dcgan)

if __name__ == '__main__':
    tf.app.run()
