import os
import re
from pathlib import Path
import numpy as np
import math
from patchify import patchify
from PIL import Image
import shutil
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
Deletes train, val, and test folders
input parameters:
    destination_dir : path to the destination directory

returns: None
"""
def delete_split_folders(destination_dir):
    folders = ['train', 'val', 'test']
    for folder in folders:
        destination_folder = os.path.join(destination_dir, folder)

        if os.path.exists(destination_folder):
          #delete folder if exists
          shutil.rmtree(destination_folder)



"""
Creates train, val, and test folders
input parameters:
    dest: path to the destination directory

returns: None
"""
def create_split_folders(destination_dir):
    folders = ['train', 'val', 'test']
    for folder in folders:
        destination_folder = os.path.join(destination_dir, folder)
        if not os.path.exists(destination_folder):
            folder_images = f"{destination_folder}/images"
            folder_masks = f"{destination_folder}/masks"
            os.makedirs(folder_images) if not os.path.exists(folder_images) else print('folder already exists')
            os.makedirs(folder_masks) if not os.path.exists(folder_masks) else print('folder already exists')

    models_folder = os.path.join(destination_dir, "models")
    if not os.path.exists(models_folder):
        os.makedirs(models_folder) if not os.path.exists(models_folder) else print('folder already exists')



"""
Splits the data into train, val, and test folders
input parameters:
    data_dir: path to the data directory
    train_ratio: ratio of the data to be used for training
    val_ratio: ratio of the data to be used for validation

returns:
    train_list: list of file names to be used for training
    val_list: list of file names to be used for validation
    test_list: list of file names to be used for testing
"""
def split_train_test_val_folders(data_dir, train_ratio, val_ratio):
  #get list of file names in the data directory
  file_list = os.listdir(data_dir)
  np.random.shuffle(file_list)


  train_size = int(len(file_list) * train_ratio)
  val_size = int(len(file_list) * val_ratio)

  train_files = file_list[:train_size]
  train_list = [x[6:10] for x in train_files]
  val_files   = file_list[train_size:train_size + val_size]
  val_list = [x[6:10] for x in val_files]
  test_files  = file_list[train_size + val_size:]
  test_list = [x[6:10] for x in test_files]
  return train_list, val_list, test_list

"""
Creates patches(smaller symmetric splits) of images
input parameters:
    src: path to the source image
    destination_path: path to the destination folder

returns: None
"""
def create_patches(src, destination_path):
    path_split = os.path.split(src)
    image = Image.open(src)
    image = np.asarray(image)
    if len(image.shape) > 2:  # only if color channel exists as well
        patches = patchify(image, (320, 320, 3), step=300)
        file_name_wo_ext = Path(src).stem
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i, j, 0]
                patch = Image.fromarray(patch)
                num = i * patches.shape[1] + j
                patch.save(f"{destination_path}/{file_name_wo_ext}_patch_{num}.png")



"""
Splits the data into train, val, and test folders
input parameters:
    src_dir: path to the source directory
    destination_dir: path to the destination directory
    train_list: list of file names to be used for training
    val_list: list of file names to be used for validation
    test_list: list of file names to be used for testing

returns: None
"""
def preprocess_data(src_dir, destination_dir, train_list, val_list, test_list):
    for path_name, _, file_name in os.walk(src_dir):
      for f in file_name:
        path_split = os.path.split(path_name)
        img_type = path_split[1]  # either 'clean which are masks' or 'render which are images'

        #get the file number
        if img_type == 'render' or img_type == 'ground':
          file_num = f[6:10]
        elif img_type == 'clean':
          file_num = f[5:9]
        else:
          print(f"unknown type {img_type}")
          return

        if file_num in val_list:
            target_folder = os.path.join(destination_dir, 'val')
        elif file_num in test_list:
            target_folder = os.path.join(destination_dir, 'test')
        elif file_num in train_list:
            target_folder = os.path.join(destination_dir, 'train')
        else:
          print(f"unknown list for {file_num}")
          return

        # copy all images
        src = os.path.join(path_name, f)

        # create patches
        if img_type == 'render':
            destination = os.path.join(target_folder, "images")
            create_patches(src=src, destination_path=destination)

        # copy all masks
        if img_type == 'clean':
            destination = os.path.join(target_folder, "masks")
            create_patches(src=src, destination_path=destination)
