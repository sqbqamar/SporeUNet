# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 06:06:51 2024

@author: drsaq
"""

import tensorflow as tf
import os
import random
import numpy as np
import glob
import pickle
from tqdm import tqdm 
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import argparse

def divide_into_patches(image, patch_size):
    height, width = image.shape[:2]
    patches = []
    
    # Calculate the number of patches in each dimension
    num_patches_x = (width + patch_size - 1) // patch_size
    num_patches_y = (height + patch_size - 1) // patch_size
    
    for y in range(num_patches_y):
        for x in range(num_patches_x):
            # Calculate the coordinates for each patch
            left = x * patch_size
            upper = y * patch_size
            right = min(left + patch_size, width)
            lower = min(upper + patch_size, height)
            
            # Crop the patch from the original image
            patch = image[upper:lower, left:right]
            patches.append(patch)
    
    return patches

def reconstruct_from_patches(patches, original_size, patch_size = 256):
    height, width = original_size
    num_patches_x = (width + patch_size - 1) // patch_size
    num_patches_y = (height + patch_size - 1) // patch_size
    
    # Create an empty image with the original size
    new_image = np.zeros((height, width), dtype=np.uint8)
    
    patch_index = 0
    for y in range(num_patches_y):
        for x in range(num_patches_x):
            # Calculate the coordinates to paste each patch
            left = x * patch_size
            upper = y * patch_size
            right = min(left + patch_size, width)
            lower = min(upper + patch_size, height)
            
            
            # Get the current patch
            patch = patches[patch_index]
            patch_height = lower - upper
            patch_width = right - left
            patch = patch[:patch_height, :patch_width]
            
            
            # Paste the patch onto the new image
            new_image[upper:lower, left:right] = patch
            
            patch_index += 1
    
    return new_image


# Load the model architecture
model = load_model("C:/Users/drsaq/OneDrive/Desktop/256_paches/model_architecture.h5")

# Load the weights
model.load_weights('C:/Users/drsaq/OneDrive/Desktop/256_paches/spore_test_dec5.hdf5')


def main():
    parser = argparse.ArgumentParser(description="Image Prediction Tool")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("output_dir", help="Directory to save the output images")
    args = parser.parse_args()

    original_img = cv2.imread(args.input_image, 0)

# Get the height, width, and number of color channels of the input image
    height, width= original_img.shape

    new_height = (height // 256) * 256 
    new_width = (width // 256) * 256


# Determine the desired image size based on the width of the input image
    if 100 < width < 512:
    

    # Resize the input image
        test_img_other = cv2.resize(original_img, (256, 256))
    
    # Extend the dimension to feed into network
        t_img_other = np.expand_dims(test_img_other, axis=-1)
        t_img_other = np.expand_dims(t_img_other, 0)

    # Make a prediction
        prediction_other = (model.predict(t_img_other)[0, :, :, 0] > 0.5).astype(np.uint8)


    else:

        large_image = cv2.resize(original_img, (new_width, new_height))
        patch_size = 256
        predicted_patches = []

    # Divide the large image into patches
        patches = divide_into_patches(large_image, patch_size)

        for patch in patches:
            patch = np.expand_dims(patch, axis=-1)
            patch = np.expand_dims(patch, 0)
            img = (model.predict(patch)[0,:,:,0] > 0.5).astype(np.uint8)
            predicted_patches.append(img)

        prediction_other = reconstruct_from_patches( predicted_patches, large_image.shape[:2], patch_size)
        
    prediction_img_path = os.path.join(args.output_dir, "prediction.jpg")
    prediction_plot_path = os.path.join(args.output_dir, "prediction_plt.jpg")
    cv2.imwrite(prediction_img_path, prediction_other)


# Plot and save the prediction
    plt.figure(figsize=(10, 10))
    plt.title('Prediction of external Image')
    plt.imshow(prediction_other, cmap='gray')
    plt.axis('off')
    plt.savefig(prediction_plot_path, dpi=700, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()