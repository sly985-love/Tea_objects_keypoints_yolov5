import os

import cv2
import numpy as np
import scipy
from PIL import Image
from matplotlib import pyplot as plt

# D_S = scipy.io.loadmat("BD04_inf_201724_004_01_DS.mat")
# D = D_S['NIR_DEPTH_res_crop'][:, :, 1:]
# S = D_S['NIR_DEPTH_res_crop'][:, :, :1]
# a = D
# print(D)
# print(S)

# # print(D)
# # print(S)
# cv2.imwrite('2.png',D)
# cv2.imwrite('1.png',S)

# path = r'D:\茶叶嫩芽3d项目\KFuji_RGB-DS_dataset\preprocessed data\mat'
# items = os.listdir(path)
# print(items)
#
# for item in items:
#     # print(item)
#     # print(path+"\\"+item)
#     patht=path + "\\" + item
#     D_S = scipy.io.loadmat(patht)
#     D = D_S['NIR_DEPTH_res_crop'][:, :, 1:]
#     S = D_S['NIR_DEPTH_res_crop'][:, :, :1]
#     filename = item.split(".")[0]
#     cv2.imwrite('depth/' + filename + '_D.png', D)
#     cv2.imwrite('D/' + filename + '_S.png', S)
#
#     BD04_inf_201724_004_01_RGBhr

path = r'D:\data'
path_new = r'D:\images'
items = os.listdir(path)
for item in items:
    if item.split("_")[-1] == "RGBhr.jpg":
        a = item.split(".")
        b = a[0][:-6] + "_DS_D." + a[1]
        patht = path + "\\" + item
        pathn = path_new + "\\" + b
        print(patht,pathn)
        src = cv2.imread(patht)
        cv2.imwrite(pathn, src)

        # print(a)
        # print(b)
        # patht = path + "\\" + item
    # a=cv2.imread(patht)
    # cv2.imwrite(patht,a)
