#########################
#                       #
#     Optical Flow      #
# (Horn–Schunck method) #
#                       #
# Author: David         #
#                       #
#########################

# reference
# https://github.com/lmiz100/Optical-flow-Horn-Schunck-method

import numpy as np
from scipy.ndimage.filters import convolve as filter2

# compute derivatives of the image intensity values along the x, y, time
def get_derivatives(img1, img2):

    # derivative masks
    kernel_x = np.array([[-1, 1], [-1, 1]]) * 0.25
    kernel_y = np.array([[-1, -1], [1, 1]]) * 0.25
    kernel_t = np.ones((2, 2)) * 0.25

    fx = filter2(img1, kernel_x) + filter2(img2, kernel_x)
    fy = filter2(img1, kernel_y) + filter2(img2, kernel_y)
    ft = filter2(img1, kernel_t) + filter2(img2, -kernel_t)

    return [fx,fy, ft]


# Horn–Schunck method of estimating optical flow
# https://en.wikipedia.org/wiki/Horn%E2%80%93Schunck_method
def computeHS(img1, img2, alpha_square, max_iterations, delta):

    # set up initial values
    u = np.zeros((img1.shape[0], img1.shape[1]))
    v = np.zeros((img1.shape[0], img1.shape[1]))
    fx, fy, ft = get_derivatives(img1, img2)
    kernel_avg = np.array([[1 / 12, 1 / 6, 1 / 12],
                            [1 / 6, 0, 1 / 6],
                            [1 / 12, 1 / 6, 1 / 12]], float)
    iter_counter = 0
    while True:
        iter_counter += 1

        # Compute local averages of the flow vectors
        u_avg = filter2(u, kernel_avg)
        v_avg = filter2(v, kernel_avg)

        # common part of update step
        p = (fx * u_avg + fy * v_avg + ft) / (alpha_square + fx**2 + fy**2)

        prev = u

        u = u_avg - fx * p
        v = v_avg - fy * p

        if iter_counter >= max_iterations:
            break
        diff = np.linalg.norm(u - prev, 2)
        if  diff < delta:
            break
    
    return [u, v]