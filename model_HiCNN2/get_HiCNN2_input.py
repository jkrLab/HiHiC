
import numpy as np
import sys
import math

sub_mat_n = 40
step = 28

# Human hg19 chromosome length
#chrs_length = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566, 155270560]

# Mouse mm9 chromosome length
#chrs_length = [195471971,182113224,160039680,156508116,151834684,149736546,145441459,129401213,124595110,130694993,122082543,120129022,120421639,124902244,104043685,98207768,94987271,90702639,61431566,171031299]


file_in_reads    = sys.argv[1]
chr_len          = int(sys.argv[2])
resolution       = int(sys.argv[3])
file_out_subMats = sys.argv[4]
file_out_index   = sys.argv[5]

num_bins = math.ceil(chr_len / resolution)
chrMat = np.zeros((num_bins, num_bins)).astype(np.int16)
# read pairs
filein = open(file_in_reads).readlines()
for i in range(0, len(filein)):
	items = filein[i].split(' ')
	i1 = math.ceil(int(items[0]) / resolution) - 1
	i2 = math.ceil(int(items[1]) / resolution) - 1 
	chrMat[i1][i2] += 1
	chrMat[i2][i1] += 1

subMats = []
index = []
for i in range(0, num_bins, step):
	for j in range(i, num_bins, step):
	  # abs(j-i)>201 means that we only care about genomic distance <= 2Mb
		if (i + sub_mat_n >= num_bins or j + sub_mat_n >= num_bins or abs(j-i) > 201):
			continue
		subMat = chrMat[i:i+sub_mat_n, j:j+sub_mat_n]
		subMats.append([subMat,])
		index.append((i+6, j+6))

index = np.array(index)
subMats = np.array(subMats)
subMats = subMats.astype(np.double)
np.save(file_out_subMats, subMats)
np.save(file_out_index, index)


