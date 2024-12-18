import os, sys
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from skimage.measure import compare_mse
from skimage.measure import compare_ssim
from DFHiC_model import DFHiC


################################################## Added by HiHiC ######
########################################################################

import argparse

parser = argparse.ArgumentParser(description='DFHiC prediction process')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')

required.add_argument('--root_dir', type=str, metavar='/HiHiC', required=True,
                      help='HiHiC directory')
required.add_argument('--model', type=str, default='DFHiC', metavar='DFHiC', required=True,
                      help='model name')
required.add_argument('--ckpt_file', type=str, metavar='[2]', required=True,
                      help='pretrained model')
required.add_argument('--batch_size', type=int, default=64, metavar='[3]', required=True,
                      help='input batch size for training (default: 64)')
required.add_argument('--gpu_id', type=int, default=0, metavar='[4]', required=True, 
                      help='GPU ID for training (defalut: 0)')
required.add_argument('--input_data', type=str, metavar='[5]', required=True,
                      help='directory path of training model')
required.add_argument('--output_data_dir', type=str, default='./output_enhanced', metavar='[6]', required=True,
                      help='directory path for saving enhanced output (default: HiHiC/output_enhanced/)')
args = parser.parse_args()

batch_size=args.batch_size
test_data = np.load(args.input_data, allow_pickle=True)['data']
num_samples = test_data.shape[0]
prefix = os.path.splitext(os.path.basename(args.input_data))[0]
os.makedirs(args.output_data_dir, exist_ok=True) #######################
########################################################################


# os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]	# -1
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
# chrome=sys.argv[2]	# 22

input_matrix = tf.placeholder('float32', [None, None, None, 1], name='matrix_input')
net = DFHiC(input_matrix, is_train=False, reuse=False)   

# test_data=np.loadtxt("GM12878/intra_LR/LR_10k_NONE.chr%s"%chrome)
input_data = np.load(args.input_data, allow_pickle=True)['inds']

# lr_data=test_data.reshape((1,test_data.shape[0],test_data.shape[1],1))
lr_data=test_data.reshape((test_data.shape[0],40,test_data.shape[1],1))

# model_path="Pretrained_weights/DFHiC_model.npz"
# model_path="check/DFHiC_best.npz"

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
tl.layers.initialize_global_variables(sess)
# tl.files.load_and_assign_npz(sess=sess, name=model_path, network=net)
tl.files.load_and_assign_npz_dict(name=args.ckpt_file, sess=sess)

# sr_matrix = sess.run(net.outputs, {input_matrix: lr_data})
##############################################################
output_batches = [] ############################# by HiHiC ###
for start in range(0, num_samples, batch_size):
    end = min(start + batch_size, num_samples)
    batch = lr_data[start:end]
    sr_batch = sess.run(net.outputs, {input_matrix: batch})
    output_batches.append(sr_batch)
sr_matrix =  np.concatenate(output_batches, axis=0)###########
##############################################################
print(sr_matrix.shape)
# result_data=sr_matrix.reshape((sr_matrix.shape[1],sr_matrix.shape[2]))
result_data = sr_matrix
print(result_data.shape)
print("***************")
# np.savetxt('DFHiC_predicted_SR_NONE_result_chr%s.txt'%chrome, result_data)
th_model = args.ckpt_file.split('/')[-1].split('_')[0]
file = os.path.join(args.output_data_dir, f'{prefix}_{args.model}_{th_model}ep.npz')
np.savez(file, data=result_data, inds=input_data)