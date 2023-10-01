#############################
#                           #
# Image Matching            #
#                           #
# Author: David Wang        #
# Created on March 03, 2023 #
#                           #
#############################

import argparse
from src.image_matching_utils import *

# ##########
# parameters 
# ##########

block_size_testing_set=[9,11,15,21,31]
window_overlapping_weight = [0.25, 0.75]
search_range = 5
window_stride = 30

# ########
# argparse 
# ########
def parse_args():
  parser = argparse.ArgumentParser(description="stereo reconstruction") 
  parser.add_argument('--image_path_1', type=str, default='data/table1.jpg', help='Your 1st image name. ')
  parser.add_argument('--image_path_2', type=str, default='data/table2.jpg', help='Your 2nd image name. ')
  parser.add_argument('--output_dir', type=str, default='results_image_matching', help='Your output directory. ') 
  return parser.parse_args() # Parse the argument 

# #####
# main
# #####
def main(): 
	args = parse_args() # Parse the argument

	img_a = io.imread(args.image_path_1, 2).astype("float")
	img_b = io.imread(args.image_path_2, 2).astype("float")
	# plot_2figures(img_a, img_b)

	for block_size in block_size_testing_set:
		print('Start to test {}x{} block'.format(block_size, block_size))
		# window_stride = block_size
		patch_list_a, position_tuple_a = get_block_position(img_a, block_size, block_size, stride=window_stride)
		patch_list_b, position_tuple_b = get_block_position(img_b, block_size, block_size, stride=1)
		motion_tuple = get_motion(patch_list_a, position_tuple_a, patch_list_b, position_tuple_b, search_range)
		X, Y, U, V = generate_2d_arrows(motion_tuple)
		plot_overlapped2images(img_a, img_b, args.output_dir, block_size, window_overlapping_weight, X, Y, U, V)

if __name__ == '__main__':
	main()