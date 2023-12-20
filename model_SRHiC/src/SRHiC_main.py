#!/usr/bin/env python
# -*- coding: utf-8 -*-

import SRHiC_predict
import SRHiC
import tensorflow as tf

################################################## Added by HiHiC ######
########################################################################
import argparse

parser = argparse.ArgumentParser(description='SRHiC training process')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('--root_dir', type=str, metavar='/HiHiC', required=True,
                      help='HiHiC directory')
required.add_argument('--model', type=str, metavar='SRHiC', required=True,
                      help='model name')
required.add_argument('--epoch', type=int, default=128, metavar='[2]', required=True,
                      help='training epoch (default: 128)')
required.add_argument('--batch_size', type=int, default=64, metavar='[3]', required=True,
                      help='input batch size for training (default: 64)')
required.add_argument('--gpu_id', type=int, default=0, metavar='[4]', required=True, 
                      help='GPU ID for training (defalut: 0)')
required.add_argument('--output_model_dir', type=str, default='./checkpoints_SRHiC', metavar='[5]', required=True,
                      help='directory path of training model (default: HiHiC/checkpoints_SRHiC/)')
required.add_argument('--loss_log_dir', type=str, default='./log', metavar='[6]', required=True,
                      help='directory path of training log (default: HiHiC/log/)')
required.add_argument('--train_data_dir', type=str, metavar='[7]', required=True,
                      help='directory path of training data')
optional.add_argument('--valid_data_dir', type=str, metavar='[8]',
                      help="directory path of validation data, but hicplus doesn't need")
args = parser.parse_args()
########################################################################
########################################################################


print(args.root_dir)

sess = tf.compat.v1.Session()

# paramers
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("iterations_size", args.epoch, " ")
tf.app.flags.DEFINE_integer("feature_size", 32, "")
tf.app.flags.DEFINE_float("epsilon",1e-6," ")

# tf.app.flags.DEFINE_string("train_input_dir","Please enter your input data folder path/"," ")
tf.app.flags.DEFINE_string("train_input_dir", args.train_data_dir," ")

# tf.app.flags.DEFINE_string("valid_input_dir", "Please enter your valid data folder path/"," ")
tf.app.flags.DEFINE_string("valid_input_dir", args.valid_data_dir," ")

# tf.app.flags.DEFINE_string("SRHiC_saver_dir","Please enter your model-saver folder path/"," ")
tf.app.flags.DEFINE_string("SRHiC_saver_dir", args.output_model_dir," ")

tf.app.flags.DEFINE_integer("GPU_ID", args.gpu_id," ")
tf.app.flags.DEFINE_integer("BATCH_SIZE", args.batch_size," ")
tf.app.flags.DEFINE_string("LOSS_LOG_DIR", args.loss_log_dir," ")

tf.app.flags.DEFINE_string("test_input_dir","Please enter your test input data folder path/"," ")
# tf.app.flags.DEFINE_string("test_input_dir",""," ")

tf.app.flags.DEFINE_string("SRHiC_checkpoint_dir","Please enter your model-saver folder path/model"," ")
# tf.app.flags.DEFINE_string("SRHiC_checkpoint_dir",OUT_DIR," ")


def main(training):
    if training:
        SRHiC.model(
            train_input_dir=FLAGS.train_input_dir,
            valid_input_dir=FLAGS.valid_input_dir,
            saver_dir=FLAGS.SRHiC_saver_dir,
            feature_size=FLAGS.feature_size,
            iterations_size=FLAGS.iterations_size,
            LOSS_LOG_DIR=FLAGS.LOSS_LOG_DIR,
            GPU_ID=FLAGS.GPU_ID,
            BATCH_SIZE=FLAGS.BATCH_SIZE
        )
    else:
        SRHiC_predict.predict(
            test_input_dir=FLAGS.test_input_dir,
            checkpoint_dir=FLAGS.SRHiC_checkpoint_dir,
            predict_save_dir=FLAGS.SRHiC_saver_dir,
        )

if __name__ == '__main__':
    main(training=True)
    pass
