###########################
#                         #
#      Optical Flow       #
#  (Horn–Schunck method)  #
#  (Lucas-Kanade method)  #
#                         #
# Author: David           #
# Created on May 14, 2023 #
#                         #
###########################

# reference
# https://github.com/lmiz100/Optical-flow-Horn-Schunck-method

import cv2
from argparse import ArgumentParser

from src.optical_flow_utils import *
from src.HornSchunckOpticalFlow import computeHS
from src.LucasKanadeOpticalFlow import ComputeLK


if __name__ == '__main__':

	# translate_image()

    parser = ArgumentParser(description = 'Horn Schunck program')
    parser.add_argument('-method', type = str, help='Type hs (Horn-Schunck) or lk (Lucas-Kanade) method for optical-flow estimation', default="hs")
    parser.add_argument('-img1', type = str, help = 'First image name (include format)', default="data/sphere1.bmp")
    parser.add_argument('-img2', type = str, help='Second image name (include format)', default="data/sphere2.bmp")
    # parser.add_argument('-img1', type = str, help = 'First image name (include format)', default="data/table1.jpg")
    # parser.add_argument('-img2', type = str, help='Second image name (include format)', default="data/table2.jpg")
    args = parser.parse_args()

    beforeImg = load_image(args.img1, '1st Image', 'results/img1.png')
    afterImg = load_image(args.img2, '2nd Image', 'results/img2.png')

    # removing noise
    kernel_blur_size = 5
    beforeImg  = cv2.GaussianBlur(beforeImg, (kernel_blur_size, kernel_blur_size), 0)
    afterImg = cv2.GaussianBlur(afterImg, (kernel_blur_size, kernel_blur_size), 0)

    if args.method == 'hs': 
        print('\n\tHorn–Schunck Optical Flow')
        Lambdas = [0.1, 1, 10]
        Iterations = [1, 4, 16, 64]
        for lambda_i in Lambdas:
            for iteration_number in Iterations:
                u, v = computeHS(beforeImg, afterImg, alpha_square = 1.0/lambda_i, max_iterations = iteration_number, delta = 10**-1)
                
                verbose = True
                p1 = QuiverPlotter(u, v, beforeImg, verbose, 'HS', lambda_i, iteration_number)

    elif args.method == 'lk': 
        print('\n\tLucas-Kanade Optical Flow')

        window_size_set = [1,3,5]

        beforeImg = np.uint8(beforeImg)
        afterImg = np.uint8(afterImg)        
        for window_size in window_size_set: 
            u, v = ComputeLK( beforeImg, afterImg, window_size, min_quality=0.001)

            verbose = True
            p1 = QuiverPlotter(u, v, beforeImg, verbose, 'LK', window_size)