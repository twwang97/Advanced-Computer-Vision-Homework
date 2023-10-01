#########################
#                       #
#     Optical Flow      #
# (Lucas-Kanade method) #
#                       #
# Author: David         #
#                       #
#########################

# reference
# https://github.com/Utkal97/Object-Tracking

import numpy as np
import cv2
from scipy.ndimage.filters import convolve as filter2

def ComputeLK(old_frame, new_frame, window_size, min_quality=0.01):

    max_corners = 10000
    min_distance = 0.1
    feature_list = cv2.goodFeaturesToTrack(old_frame, max_corners, min_quality, min_distance)
    print(feature_list.shape)

    w = int(window_size/2)

    old_frame = old_frame / 255.0
    new_frame = new_frame / 255.0

    # Convolve to get gradients w.r.t. X, Y and T dimensions
    kernel_x = np.array([[-1, 1], [-1, 1]]) * 0.25
    kernel_y = np.array([[-1, -1], [1, 1]]) * 0.25
    kernel_t = np.ones((2, 2)) * 0.25

    fx = filter2(old_frame, kernel_x) # Gradient over X
    fy = filter2(old_frame, kernel_y) # Gradient over Y
    ft = filter2(old_frame, -kernel_t) + filter2(new_frame, kernel_t) # Gradient over Time

    u = np.zeros(old_frame.shape)
    v = np.zeros(old_frame.shape)

    for feature_i in feature_list:        #   for every corner
            j, i = feature_i.ravel()      #   get cordinates of the corners (i,j). They are stored in the order j, i
            i, j = int(i), int(j)       #   i,j are floats initially

            I_x = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            I_y = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            I_t = ft[i-w:i+w+1, j-w:j+w+1].flatten()

            b = np.reshape(I_t, (I_t.shape[0],1))
            A = np.vstack((I_x, I_y)).T

            U = np.matmul(np.linalg.pinv(A), b)

            u[i,j] = U[0][0]
            v[i,j] = U[1][0]
 
    return (u,v)