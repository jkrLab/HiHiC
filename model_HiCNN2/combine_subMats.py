
import sys
import numpy as np


dat_predict = np.squeeze(np.load(sys.argv[1]).astype(np.float32))
dat_index   = np.load(sys.argv[2])
chr_len     = int(sys.argv[3])
resolution  = int(sys.argv[4])
file_output = sys.argv[5]

num_bins = np.ceil(chr_len / resolution).astype('int')
mat = np.zeros((num_bins, num_bins))

for i in range(dat_predict.shape[0]):
	r1 = dat_index[i,0]
	c1 = dat_index[i,1]
	r2 = r1 + 27 + 1
	c2 = c1 + 27 + 1
	mat[r1:r2, c1:c2] = dat_predict[i,:,:]


# copy upper triangle to lower triangle
lower_index = np.tril_indices(num_bins, -1)
mat[lower_index] = mat.T[lower_index]  


np.save(file_output, mat)


