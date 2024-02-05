# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 05:34:36 2024

@author: drsaq
"""

import cv2
import numpy as np

# Load TIF image 
tif_image = cv2.imread('input_image.tif')

# OpenCV reads TIF images as BGR format by default


# OpenCV saves images in BGR format by default
# So TIF image can be saved directly as JPG without any conversion

jpg_path = 'output_image.jpg'
cv2.imwrite(jpg_path, tif_image)

# Read back the saved jpg
jpg_image = cv2.imread(jpg_path) 

if jpg_image is not None:
  print('TIF image successfully converted and saved as JPG')