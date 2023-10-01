#########################
#                       #
#     Optical Flow      #
#                       #
# Author: David         #
#                       #
#########################

import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image(img_path, img_title, img_output_path):

    img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float)

    ax = plt.figure().gca()
    ax.imshow(img1, cmap = 'gray')
    ax.set_title(img_title)
    ax.axis((0, img1.shape[1], img1.shape[0], 0))
    fig1 = plt.gcf()
    plt.show()
    plt.draw()

    fig1.savefig(img_output_path)
    print(img_output_path, 'is saved')

    return img1

def translate_image(): 
    # HW3: Get another translated image
    lena = cv2.imread("./lena.bmp", 0)

    # padding
    lena_ = np.hstack((np.zeros((512,1)),lena))  
    lena_ = np.vstack((np.zeros((1,513)),lena_)) 

    # translate lena.bmp one pixel to the right and downward.
    lena_shift = lena_[:512,:512]
    print(type(lena), type(lena_shift))
    cv2.imwrite('./lena2.bmp', lena_shift)
    cv2.imshow("image", lena_shift)
    cv2.waitKey()

class QuiverPlotter:
    def __init__(self, u, v, img1, verbose, flow_method, param1, param2=None):
        self.scale = 10
        self.search_direction = 8

        if flow_method == 'HS':
            self.draw_quiver_hs(u, v, img1, param1, param2, verbose)
        elif flow_method == 'LK':
            self.draw_quiver_lk(u, v, img1, param1, verbose)

    # compute magnitude in each 8 pixels. return magnitude average
    def get_magnitude(self, u, v):
        mag_avg = 0.0
        counter = 0.0

        for i in range(0, u.shape[0], self.search_direction):
            for j in range(0, u.shape[1],self.search_direction):
                counter += 1
                dy = v[i,j]
                dx = u[i,j]
                magnitude = (dx**2 + dy**2)**0.5
                mag_avg += magnitude

        mag_avg = mag_avg / counter
        return mag_avg

    def draw_quiver_hs(self, u, v, img1, lambda_i, iteration_number, verbose):
        ax = plt.figure().gca()
        ax.imshow(img1, cmap = 'gray')

        img_output_path = 'results/lambda_' + str(lambda_i) + '_i' + str(iteration_number) + '.png'
        if iteration_number <= 1: 
            ax.set_title("Optical Flow (Lambda = {} with {} iteration)".format(lambda_i, iteration_number))
        else:
            ax.set_title("Optical Flow (Lambda = {} with {} iterations)".format(lambda_i, iteration_number))

        magnitudeAvg = self.get_magnitude(u, v)

        for i in range(0, u.shape[0], 8):
            for j in range(0, u.shape[1],8):
                dy = v[i,j]
                dx = u[i,j]
                magnitude = (dx**2 + dy**2)**0.5
                #draw only significant changes
                if magnitude > magnitudeAvg:
                    dx *= self.scale
                    dy *= self.scale
                    ax.arrow(j, i, dx, dy, color = 'red', 
                        head_width=5, head_length=2, width=0.001)

        ax.axis((0, img1.shape[1], img1.shape[0], 0))

        if verbose: 
            fig1 = plt.gcf()
            plt.show()
            plt.draw()
            
            fig1.savefig(img_output_path)
            print(img_output_path, 'is saved')

    def draw_quiver_lk(self, u, v, img1, window_size, verbose):
        ax = plt.figure().gca()
        ax.imshow(img1, cmap = 'gray')

        img_output_path = 'results/w_' + str(window_size) + '.png'

        ax.set_title("Lucas-Kanade Optical Flow (Kernel Size = {} x {})".format(window_size, window_size))
        magnitudeAvg = self.get_magnitude(u, v)

        for i in range(0, u.shape[0], 8):
            for j in range(0, u.shape[1],8):
                dy = v[i,j]
                dx = u[i,j]
                magnitude = (dx**2 + dy**2)**0.5
                #draw only significant changes
                if magnitude > magnitudeAvg:
                    dx *= self.scale
                    dy *= self.scale
                    ax.arrow(j, i, dx, dy, color = 'red', 
                        head_width=5, head_length=2, width=0.001)

        ax.axis((0, img1.shape[1], img1.shape[0], 0))

        if verbose: 
            fig1 = plt.gcf()
            plt.show()
            plt.draw()
            
            fig1.savefig(img_output_path)
            print(img_output_path, 'is saved')