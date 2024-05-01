# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:05:32 2024

@author: drsaq
"""

from flask import Flask, render_template, Response, request, send_file
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
from datetime import datetime
import shutil
from flask import current_app

#from segment import measure

# Create flask app
app = Flask(__name__)


# Load the model architecture
model = load_model("C:/Users/drsaq/OneDrive/Desktop/256_paches/model_architecture.h5")

# Load the weights
model.load_weights('C:/Users/drsaq/OneDrive/Desktop/256_paches/spore_test_dec5.hdf5')

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
def segment(input_path):
  #  parser = argparse.ArgumentParser(description="Image Prediction Tool")
  #  parser.add_argument("input_image", help="Path to the input image")
  #  parser.add_argument("output_dir", help="Directory to save the output images")
   # args = parser.parse_args()

    original_img = cv2.imread(input_path, 0)

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
        
    #prediction_img_path = os.path.join(args.output_dir, "prediction.jpg")
    #prediction_plot_path = os.path.join(args.output_dir, "prediction_plt.jpg")
    #cv2.imwrite(prediction_img_path, prediction_other)
    return prediction_other

def find_most_recent_folder(directory):
    folders = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    if not folders:
        return None
    most_recent_folder = max(folders, key=os.path.getctime)

    # Remove all other folders except the most recent one
    for folder in folders:
        if folder != most_recent_folder:
            try:
                shutil.rmtree(folder)
            except OSError:
                pass

    return most_recent_folder

def find_most_recent_image_in_folder(folder):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    recent_image = None
    recent_time = 0

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                file_path = os.path.join(root, file)
                file_time = os.path.getctime(file_path)
                if file_time > recent_time:
                    recent_time = file_time
                    recent_image = file_path

    return recent_image

image_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.tif')
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload')

@app.route('/', methods=['GET', 'POST'])
def application():
    if request.method == 'POST':
        # Take uploaded image
        upload_file = request.files['image_name']
        nowTime = datetime.now().strftime("%Y%m%d%H%M%S")
        
        filename = nowTime + '_' + upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        # Store image in upload directory
        upload_file.save(path_save)
        image_base_name = os.path.splitext(os.path.basename(path_save))[0]
        # Take image and perform OCR
        img = cv2.imread(path_save)
        if path_save:
            # Now, delete other images
            for root, dirs, files in os.walk(UPLOAD_PATH):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file_path != path_save and file.lower().endswith(image_extensions):
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass
            
        img = segment(path_save)
       # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        plt.figure(figsize=(6, 6))
        #plt.title('Prediction of Test Image')
        plt.imshow(img, cmap='gray')
        plt.axis('off')  # Turn off axis labels
        plt.savefig('static/Result/output.jpg', dpi=500, bbox_inches='tight')

       # cv2.imwrite('static/Result/output.jpg', img)
        directory_path = 'static/Result' # Replace with the actual directory path
        #recent_folder = find_most_recent_folder(directory_path)
        image_path = find_most_recent_image_in_folder(directory_path)
       # image1_path = 'static/prediction/image0.jpg'
       # image1_name = os.path.basename(image_path)

        return render_template('index.html', upload = True, upload_image = filename, image1 = image_path)

    return render_template('index.html', upload = False)

if __name__ == "__main__":
    #app.run()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    
    
    