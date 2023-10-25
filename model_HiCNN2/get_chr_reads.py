#!/usr/bin/env python3

import sys

file_HiC   = sys.argv[1]
chr        = sys.argv[2]
file_reads = sys.argv[3]


fileIn  = open(file_HiC)
fileOut = open(file_reads, "w")

for line in fileIn:
	items = line.split()
	if len(items) != 11:
		continue
	if items[2] == chr and items[6] == chr:
		outstr = items[3] + " " + items[7] + "\n"
		fileOut.write(outstr)

fileIn.close()
fileOut.close()


