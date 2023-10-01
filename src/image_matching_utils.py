#############################
#                           #
# Utils for                 #
#      Image Matching       #
#                           #
# Author: David Wang        #
# Created on March 03, 2023 #
#                           #
#############################


from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def get_block_position(img, windowsize_r, windowsize_c, stride):
    patch_list = []
    position_tuple = []
    # Crop your picture
    for r in range(0, img.shape[0] - windowsize_r, stride):
        for c in range(0, img.shape[1] - windowsize_c, stride):
            position_tuple.append((r,c))
            window = img[r:r+windowsize_r,c:c+windowsize_c]
            patch_list.append(window)
    return patch_list, position_tuple

def get_motion(patch_list_a, position_tuple_a, patch_list_b, position_tuple_b, search_range=50):
    motion_tuple = []
    for patch_a, position_a in zip(patch_list_a, position_tuple_a):
        cost = 999999
        for patch_b, position_b in zip(patch_list_b, position_tuple_b):
            position_error = ((position_a[0]-position_b[0])**2 + (position_a[1]-position_b[1])**2)**0.5
            if position_error <= search_range:
                difference = np.sum(abs(patch_a-patch_b))
                if difference <= cost:
                    cost = difference
                    position_best = position_b
        dr, dc = position_best[0]-position_a[0], position_best[1] - position_a[1]
        motion_tuple.append((position_a, (dr, dc)))
    return motion_tuple

def plot_2figures(img1, img2):
	plt.figure()
	plt.subplot(2,2,1)
	plt.imshow(img1, cmap="gray")
	plt.subplot(2,2,2)
	plt.imshow(img2, cmap="gray")
	plt.tight_layout()
	plt.show()

def plot_overlapped2images(img1, img2, output_dir, block_size, window_overlapping_weight, X, Y, U, V):
	plt.figure()
	plt.imshow(window_overlapping_weight[0] * img1 + window_overlapping_weight[1] * img2, cmap="gray")
	plt.quiver(X, Y, U, V, color='r')
	plt.title("Motion Vector for {}x{} Block".format(block_size, block_size))
	img_name = "./{}/motion_with_block_{}.png".format(str(output_dir), str(block_size))
	print("Output: ", img_name)
	plt.savefig(img_name, transparent=False)
	plt.show()

def generate_2d_arrows(motion_tuple):
	X = [t[0][1] for t in motion_tuple]
	Y = [t[0][0] for t in motion_tuple]
	U = [t[1][1] for t in motion_tuple]
	V = [-t[1][0] for t in motion_tuple]
	return X, Y, U, V