############################
#                          #
# Stereo Reconstruction    #
#                          #
# Author: David Wang       #
# Created on Oct. 01, 2023 #
#                          #
############################

import numpy as np
from scipy import ndimage # median_filter
import cv2 as cv
import matplotlib.pyplot as plt
import open3d as o3d
import argparse

from src.read_stereo_txt import stereo_config_info

# datasets: 
# https://vision.middlebury.edu/stereo/data/scenes2014/

def find_disparity(imgL, imgR, camera_config_path, disparity_output_path=None):
    Dic,k,max_disparity,min_disparity,num_disparities,window_size,H,W,Bs,dist = stereo_config_info(camera_config_path)
    T = np.zeros((3,1))
    T[0,0] = Bs
    _,_,_,_,Q,_,_ = cv.stereoRectify(k, dist, k, dist, (H, W), np.identity(3), T)
    stereo = cv.StereoSGBM_create(minDisparity = min_disparity, 
                                  numDisparities = num_disparities, 
                                  preFilterCap = 1, 
                                  blockSize = 5, 
                                  uniquenessRatio = 2, 
                                  speckleWindowSize = 50, 
                                  speckleRange = 2, 
                                  disp12MaxDiff = 1, 
                                  P1 = 8*3*window_size**2, 
                                  P2 = 32*3*window_size**2,
                                  mode = 4)
    
    disparity = stereo.compute(imgL,imgR).astype(np.float32)
    disparity = ndimage.median_filter(disparity, size=21)

    if disparity_output_path is not None: 
      plt.imshow(disparity,"jet")
      # plt.show()
      plt.savefig(disparity_output_path)

    return disparity, Q

def disparity_to_pointcloud(disparity, Q, img1):
  # To reproject a disparity image to 3D space.
  point_cloud = cv.reprojectImageTo3D(disparity,Q) 
  mask = disparity > disparity.min()
  rgb_points = img1[mask]
  rgb_points = rgb_points.reshape(-1,3)
  xyz_points = (point_cloud[mask]).reshape(-1, 3)

  return xyz_points, rgb_points

def write_pointcloud(filename, xyz_points, rgb_points=None):
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_points)
    if rgb_points is not None:
      rgb_points[:, [2, 0]] = rgb_points[:, [0, 2]] # for Open3D format
      pcd.colors = o3d.utility.Vector3dVector(rgb_points/255.)
    o3d.visualization.draw_geometries([pcd])
    print("point cloud is written as ", filename)
    o3d.io.write_point_cloud(filename, pcd)

def parse_args():
  parser = argparse.ArgumentParser(description="stereo reconstruction") 
  parser.add_argument('--dir', type=str, default='data/motorcycle', help='Your input directory. ')
  parser.add_argument('--image_1_path', type=str, default='img0.png', help='Your 1st image path. ')
  parser.add_argument('--image_2_path', type=str, default='img1.png', help='Your 2nd image path. ')
  parser.add_argument('--calibration_info_path', type=str, default='calibration_info.txt', help='Your calibration information. ')
  parser.add_argument('--disparity_path', type=str, default='disparity.jpg', help='Your disparity image path. ')
  parser.add_argument('--output_path', type=str, default='output.ply', help='Your output *.ply name. ') 
  args = parser.parse_args() # Parse the argument
  args.image_1_path = args.dir + '/' + args.image_1_path
  args.image_2_path = args.dir + '/' + args.image_2_path
  args.calibration_info_path = args.dir + '/' + args.calibration_info_path
  args.disparity_path = args.dir + '/' + args.disparity_path
  args.output_path = args.dir + '/' + args.output_path
  return args 

if __name__ == "__main__":
  args = parse_args() # Parse the argument
  imageLeft = cv.imread(args.image_1_path)
  image_gray_left = cv.imread(args.image_1_path, 0)
  image_gray_right = cv.imread(args.image_2_path, 0)
  print(image_gray_left.shape, image_gray_right.shape)
  disparity,Q = find_disparity(image_gray_left, image_gray_right, args.calibration_info_path, args.disparity_path)
  point_cloud,color = disparity_to_pointcloud(disparity, Q, imageLeft)
  write_pointcloud(args.output_path,point_cloud,color)