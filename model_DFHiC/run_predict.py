import os, sys
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from skimage.measure import compare_mse
from skimage.measure import compare_ssim
from DFHiC_model import DFHiC

###########################################################
root_dir = "/data/HiHiC-main"
cell_line = "GM12878"
low_res = "16"
ckpt_file = "/data/HiHiC-main/checkpoints_DFHiC/00190_1.02.43_7.627.npz"
cuda = "0"
# device = "cpu"
# model = "HiCARN_1"
input_file = "/data/HiHiC-main/data_DFHiC/test/test_ratio16.npz"
input_data = np.load(input_file, allow_pickle=True)
out_dir = "/data/HiHiC-main/output"
os.makedirs(out_dir, exist_ok=True)
###########################################################

# os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]	# -1
os.environ["CUDA_VISIBLE_DEVICES"] = cuda
# chrome=sys.argv[2]	# 22

input_matrix = tf.placeholder('float32', [None, None, None, 1], name='matrix_input')
net = DFHiC(input_matrix, is_train=False, reuse=False)   

# test_data=np.loadtxt("GM12878/intra_LR/LR_10k_NONE.chr%s"%chrome)
test_data = input_data['data']
print(test_data)
print(test_data.shape)

# lr_data=test_data.reshape((1,test_data.shape[0],test_data.shape[1],1))
lr_data=test_data.reshape((test_data.shape[0],40,test_data.shape[1],1))
print(lr_data.shape)

# model_path="Pretrained_weights/DFHiC_model.npz"
# model_path="check/DFHiC_best.npz"

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
tl.layers.initialize_global_variables(sess)
# tl.files.load_and_assign_npz(sess=sess, name=model_path, network=net)
tl.files.load_and_assign_npz_dict(name=ckpt_file, sess=sess)
sr_matrix = sess.run(net.outputs, {input_matrix: lr_data})
print(sr_matrix.shape)
# result_data=sr_matrix.reshape((sr_matrix.shape[1],sr_matrix.shape[2]))
result_data = sr_matrix
print(result_data.shape)
print("***************")
# np.savetxt('DFHiC_predicted_SR_NONE_result_chr%s.txt'%chrome, result_data)
np.savez(os.path.join(out_dir, f'DFHiC_predict_{low_res}'), data=result_data)