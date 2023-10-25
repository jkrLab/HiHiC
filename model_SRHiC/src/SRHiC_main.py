#!/usr/bin/env python
# -*- coding: utf-8 -*-

import SRHiC_predict
import SRHiC
import tensorflow as tf



sess = tf.compat.v1.Session()

# paramers
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("iterations_size", 200, " ")
tf.app.flags.DEFINE_integer("feature_size", 32, "")
tf.app.flags.DEFINE_float("epsilon",1e-6," ")

# tf.app.flags.DEFINE_string("train_input_dir","Please enter your input data folder path/"," ")
tf.app.flags.DEFINE_string("train_input_dir","/data/mohyelim7/intergrate_hihic_data/SRHiC/train_data_raw_ratio16.npy"," ")

# tf.app.flags.DEFINE_string("valid_input_dir", "Please enter your valid data folder path/"," ")
tf.app.flags.DEFINE_string("valid_input_dir", "/data/mohyelim7/intergrate_hihic_data/SRHiC/valid_data_raw_ratio16.npy"," ")

# tf.app.flags.DEFINE_string("SRHiC_saver_dir","Please enter your model-saver folder path/"," ")
tf.app.flags.DEFINE_string("SRHiC_saver_dir","/data/mohyelim7/SRHiC-master/model/"," ")


# # tf.app.flags.DEFINE_string("test_input_dir","Please enter your test input data folder path/"," ")
# tf.app.flags.DEFINE_string("test_input_dir","/data/mohyelim7/SRHiC-master/data/GM12878_combined_10_kb_16_chr21_test.npy"," ")

# tf.app.flags.DEFINE_string("SRHiC_checkpoint_dir","Please enter your model-saver folder path/model"," ")
tf.app.flags.DEFINE_string("SRHiC_checkpoint_dir","/data/mohyelim7/SRHiC-master/model/"," ")





def main(training):
    if training:
        SRHiC.model(
            train_input_dir=FLAGS.train_input_dir,
            saver_dir=FLAGS.SRHiC_saver_dir,
            valid_input_dir=FLAGS.valid_input_dir,
            feature_size=FLAGS.feature_size,
            iterations_size=FLAGS.iterations_size,
        )
    else:
        SRHiC_predict.predict(
            test_input_dir=FLAGS.test_input_dir,
            checkpoint_dir=FLAGS.SRHiC_checkpoint_dir,
            predict_save_dir=FLAGS.SRHiC_saver_dir,
        )

if __name__ == '__main__':
    # main(training=False)
    main(training=True)
    pass