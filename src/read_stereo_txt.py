#############################
#                           #
# Read the parameters       #
#          from txt         #
#                           #
# Author: David Wang        #
# Modified on Oct. 01, 2023 #
# Forked from               #
#     sushlokshah's Github  #
#                           #
#############################

import numpy as np

def stereo_config_info(path):
  file1 = open(path, 'r')
  Lines = file1.readlines()
  Dic = {}
  for l in Lines:
    a = l.split("=")
    if(a[1][-1]== "\n"):
      try:
        Dic[a[0]] = int(a[1][:-1])
      except:
        Dic[a[0]] = a[1][:-1]
    else:
      try:
        Dic[a[0]] = int(a[1])
      except:
        Dic[a[0]] = a[1]
  Dic["cam0"] = Dic["cam0"].split(" ")
  Dic["cam0"][0] = Dic["cam0"][0][1:]
  Dic["cam0"][2] = Dic["cam0"][2][:-1]
  Dic["cam0"][5] = Dic["cam0"][5][:-1]
  Dic["cam0"][-1] = Dic["cam0"][-1][:-1]
  Dic["cam1"] = Dic["cam1"].split(" ")
  Dic["cam1"][0] = Dic["cam1"][0][1:]
  Dic["cam1"][2] = Dic["cam1"][2][:-1]
  Dic["cam1"][5] = Dic["cam1"][5][:-1]
  Dic["cam1"][-1] = Dic["cam1"][-1][:-1]
  for i in range(len(Dic["cam0"])):
    Dic["cam0"][i] = float(Dic["cam0"][i])
    Dic["cam1"][i] = float(Dic["cam1"][i])
  Dic["cam0"] = np.array(Dic["cam0"]).reshape(3,3)
  Dic["cam1"] = np.array(Dic["cam1"]).reshape(3,3)
  max_disparity = Dic["vmax"]
  min_disparity = Dic["vmin"]
  num_disparities = max_disparity - min_disparity
  window_size = 5
  k = Dic["cam0"]
  distortion = np.zeros((5,1)).astype(np.float32)
  return Dic,k,max_disparity,min_disparity,num_disparities,window_size,Dic["height"],Dic["width"],Dic["baseline"],distortion
