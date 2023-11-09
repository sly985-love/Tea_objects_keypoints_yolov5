#!/usr/bin/env python
# encoding: utf-8

import glob
import os
import numpy as np
from PIL import Image
from shutil import copy2

outDir = os.path.abspath(r'D:\yolotea6\new_data\test')

path = r'D:\yolotea6\new_data\new_tea\val'
path_new = r'D:\yolotea6\new_data\out_dir2'
path_end = r"D:\yolotea6\new_data\val"
items1 = os.listdir(path)
items2 = os.listdir(path_new)

for item1 in items1:
    for item2 in items2:
        if item1 == item2:
            patht = path_new + "\\" + item1
            print(patht)
            copy2(patht, path_end)

#         # Use the function: os.path.join
# imageDir1 = os.path.abspath(r'D:\yolotea6\new_data\new_tea\test')
#
# # Define the List of the images
# image1 = []
#
# # Get the absolute path of the images
# imageList1 = glob.glob(os.path.join(imageDir1, '*.txt'))
#
# # Use the function: os.path.basename() Get the name of the images
# for item in imageList1:
#     image1.append(os.path.basename(item))
#
# imageDir2 = os.path.abspath(r'/D:\yolotea6\new_data\out_dir2')
# image2 = []
# imageList2 = glob.glob(os.path.join(imageDir2, '*.txt'))
#
# for item in imageList2:
#     image2.append(os.path.basename(item))
#
# for item in image1:
#     print(item)
#
# for item in image2:
#     print(item)
#
# for item1 in image1:
#     for item2 in image2:
#         if item1 == item2:
#             copy2(os.path.join(imageDir2, item1), os.path.join(outDir, item2))
#             # img = Image.open(os.path.join(imageDir2, item1))
#             # img.save(os.path.join(outDir, item2))
