# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:04:14 2024

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


# Load the model architecture
model = load_model("C:/Users/drsaq/OneDrive/Desktop/256_paches/model_architecture.h5")

# Load the weights
model.load_weights('C:/Users/drsaq/OneDrive/Desktop/256_paches/spore_test_dec5.hdf5')



# Load the test image
test_img_other = cv2.imread('C:/Users/drsaq/OneDrive/Desktop/256_paches/test/images/7.jpg', 0)
t_img_other = np.expand_dims(test_img_other, axis=-1)
t_img_other = np.expand_dims(t_img_other, 0)

# Make a prediction
prediction_other = (model.predict(t_img_other)[0, :, :, 0] > 0.5).astype(np.uint8)

# Plot and save the prediction
plt.figure(figsize=(8, 8))
#plt.title('Prediction of Test Image')
plt.imshow(prediction_other, cmap='gray')
plt.axis('off')  # Turn off axis labels
plt.savefig('C:/Users/drsaq/OneDrive/Desktop/256_paches/test/result/7.png', dpi=500, bbox_inches='tight')

# Show the plot
plt.show()



