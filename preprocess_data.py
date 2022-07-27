import re
import os
import string
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import PIL
import seaborn as sn
import tensorflow as tf
from keras.preprocessing import image
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# FInd directory
def get_dir(base_folder,train_folder):
    base_dir = os.path.join(os.getcwd(), base_folder)
    train_dir = os.path.join(base_folder, train_folder)
    return train_dir

#Load and split dataset
def load_ds(path,subset, val_split, img_h, img_w, batch_size):
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=val_split,
        subset=subset,
        seed=123,
        image_size=(img_h, img_w),
        batch_size=batch_size)
    return ds

# Show data
def show_data(ds,col,row):
    class_names = ds.class_names
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(len(class_names)):
            ax = plt.subplot(col, row, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")