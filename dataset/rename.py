import glob
import os

fname = glob.glob('./*/*.jpg')
for i in range(len(fname)):
	dest = './clean/' + str(i) + '.jpg'
	os.rename(fname[i],dest)