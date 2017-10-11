# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 21:49:26 2017

@author: yang
"""

import os
import cv2
import utils
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

def thresholding(img):
    x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=35, thresh_max=100)
    mag_thresh = utils.mag_thresh(img, sobel_kernel=9, mag_thresh=(50, 100))
    dir_thresh = utils.dir_threshold(img, sobel_kernel=21, thresh=(0.7, 1.3))
    s_thresh = utils.hls_select(img,channel='s',thresh=(180, 255))
  
    threshholded = np.zeros_like(x_thresh)
    threshholded[(x_thresh==1) | ((mag_thresh == 1) & (dir_thresh == 1)) | (s_thresh==1)] = 1
    
    return threshholded

def processing(img,object_points,img_points,M,Minv):
    undist = utils.cal_undistort(img,object_points,img_points)
    thresholded = thresholding(undist)
    thresholded_wraped = cv2.warpPerspective(thresholded, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line(thresholded_wraped)
    print(left_fit)
    result = utils.draw_area(undist,thresholded_wraped,Minv,left_fit, right_fit)
    return result

cal_imgs = utils.get_images_by_dir('camera_cal')
object_points,img_points = utils.calibrate(cal_imgs,grid=(9,6))
M,Minv = utils.get_M_Minv()

#project_outpath = 'vedio_out/project_video_out.mp4'
#project_video_clip = VideoFileClip("project_video.mp4")
#project_video_out_clip = project_video_clip.fl_image(lambda clip:processing(clip,object_points,img_points,M,Minv))
#project_video_out_clip.write_videofile(project_outpath, audio=False)
#

project_outpath = 'vedio_out/test.mp4'
project_video_clip = VideoFileClip("project_video.mp4").subclip(0,5)
project_video_out_clip = project_video_clip.fl_image(lambda clip:processing(clip,object_points,img_points,M,Minv))
project_video_out_clip.write_videofile(project_outpath, audio=False)


    