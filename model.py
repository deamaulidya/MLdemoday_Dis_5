# %%
import cv2
import os
import string
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from preprocess_data import *

# %%
# Model
def create_model(img_h,img_w,color_type):
    model = tf.keras.models.Sequential([
      # First convolution layer
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape = (img_h, img_w, color_type)),
      tf.keras.layers.MaxPooling2D(2, 2),
      # Second convolution layer
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2, 2),
      # Third convolution layer
      tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2, 2),
      # Fourth convolution layer
      tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2, 2),
      
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(26, activation='softmax')])
    return model

# %%
# Create and compile model
def create_mod(img_h,img_w,color_type):
  model = create_model(img_h, img_w, color_type)
  model.summary()
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return model

# %%
# Fit model
def fit_model(model,train_data,val_data,n_epochs, batch_size):
  result = model.fit(train_data,
                   validation_data = val_data,
                   epochs = n_epochs, batch_size = batch_size, verbose = 1)
  return result

