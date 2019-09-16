import cv2
import numpy as np
import glob


# def resize2SquareKeepingAspectRation(img, size, interpolation):
#   h, w = img.shape[:2]
#   c = None if len(img.shape) < 3 else img.shape[2]
#   if h == w: return cv2.resize(img, (size, size), interpolation)
#   if h > w: dif = h
#   else:     dif = w
#   x_pos = int((dif - w)/2.)
#   y_pos = int((dif - h)/2.)
#   if c is None:
#     mask = np.zeros((dif, dif), dtype=img.dtype)
#     mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
#   else:
#     mask = np.zeros((dif, dif, c), dtype=img.dtype)
#     mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
#   return cv2.resize(mask, (size, size), interpolation)

# img = cv2.imread('./clean/46.jpg')
# size = 224
# resized = resize2SquareKeepingAspectRation(img, size, cv2.INTER_AREA)
# cv2.imwrite('./46.jpg',resized)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized



# print(img.shape[0], img.shape[1])

def process(img):
	if img.shape[0] > img.shape[1]:
		col_target = 480
		row_target = 640
	elif img.shape[0] <= img.shape[1]:
		col_target = 640
		row_target = 480

	ratio_target = col_target/row_target
	ratio_source = img.shape[1]/img.shape[0]
	# print(ratio_target)
	# print(ratio_source)

	if ratio_source <= ratio_target:
		resized_img = image_resize(img.copy(),height=row_target)
		new_img = np.zeros((row_target,col_target,3), dtype = img.dtype)
		diff = col_target - resized_img.shape[1]
		new_img[:, int(diff/2):int(diff/2)+resized_img.shape[1], :] = resized_img
	elif ratio_source > ratio_target:
		resized_img = image_resize(img.copy(),width=col_target)
		new_img = np.zeros((row_target,col_target,3), dtype = img.dtype)
		diff = row_target - resized_img.shape[0]
		new_img[int(diff/2):int(diff/2)+resized_img.shape[0], :, :] = resized_img

	return new_img



fname = glob.glob('./clean/*.jpg')
for i in range(len(fname)):
	target_fname = fname[i].split('/')[-1]
	target_fname = './clean_resized/' + target_fname
	img = cv2.imread(fname[i])
	processed_img = process(img)
	cv2.imwrite(target_fname, processed_img)

