import glob
import cv2

f = glob.glob('./clean/*.jpg')
landscape = 0
potrait = 0
equal = 0
for i in range(len(f)):
	shape = cv2.imread(f[i]).shape
	if shape[0] < shape[1]:
		landscape += 1
	elif shape[0] > shape[1]:
		potrait += 1
	elif shape[0] == shape[1]:
		equal += 1

print('landscape: ', landscape)
print('potrait ', potrait)
print('equal ', equal)