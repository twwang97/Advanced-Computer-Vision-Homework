# Advanced Computer Vision: Homework

## Contents
### [Project 1: Image Matching](#project-1-image-matching-1)
* keywords: Block Matching
### [Project 2: Optical Flow](#project-2-optical-flow-1)
* keywords: Horn–Schunck, Lucas-Kanade
### [Project 3: Stereo Reconstruction](#project-3-stereo-reconstruction-1)
* keywords: stereo camera, image rectification, disparity map, point clouds
### Final Project
* Title: Improved Real-Time Pipeline for 3D Mesh Reconstruction from RGB-D Cameras
* external link: [abstract](https://twwang97.github.io/publication/2009-10-01-paper-title-number-1), [video](https://youtu.be/tccfjy2j27w)
### Note
* [References](#note-1) are listed at the bottom of this page.
* The pdf file ```MyNotes.pdf``` includes the material for preparing my midterm and final exams in the class of Advanced Computer Vision. These notes are provided for personal use only and may not be used for commercial purposes. If you notice any typos or have inquiries regarding the use of these notes, feel free to contact me. Thank you!


---
# Project 1: Image Matching
### Python Usage
```
python3 image_matching.py
```
![animation_motion_with_block](results_image_matching/animation_motion_with_block.gif)

---
# Project 2: Optical Flow
### Python Usage
```
python3 optical_flow.py
```
### Results from Horn–Schunck optical flow: <br>
![sphere_hs_flow](results_optical_flow/sphere/hs_flow.gif)
![table_hs_flow](results_optical_flow/table/flow.gif)

---

### Results from Lucas-Kanade optical flow: <br>
![sphere_lk_flow](results_optical_flow/sphere/lk_flow.gif)

---
# Project 3: Stereo Reconstruction
### Python Usage
```
python3 stereo_reconstruction.py
```
### Result (Demonstration [Video](https://youtu.be/aQThOlFRNxc))
![stereo_reconstruction_open3d](results_stereo_reconstruction/stereo_open3d.gif)

---
# Note
* All output files are saved in the `results` folder. 
* Some projects are modified from the class of Advanced Computer Vision at National Taiwan University.

---
# Resources and References
[1] Lecture Notes from Chiou-Shann Fuh's class of Advanced Computer Vision (Spring 2023) <br>
[2] optical flow from lmiz100's [github](https://github.com/lmiz100/Optical-flow-Horn-Schunck-method) <br>
[3] stereo algorithm from sushlokshah's [github](https://github.com/sushlokshah/stereo-reconstruction) <br>
[4] stereo [datasets](https://vision.middlebury.edu/stereo/data/scenes2014/) <br>
