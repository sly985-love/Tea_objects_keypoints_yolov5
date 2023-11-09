import os
# 画RGB彩色直方图
import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('D:\yolotea6\\all\zi1\IMG_5863.jpg')
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()
# # 6144 6160 D:\yolotea6\all\lv1
# D:\yolotea6\all\zi2
# IMG_7948


# 使用laplacian原理钝化掩蔽和高提升滤波
import cv2 as cv
import numpy as np


def clear_fig(src):
    down_dst1 = cv.pyrDown(src)
    up_dst1 = cv.pyrUp(down_dst1)
    laplace1 = cv.subtract(src, up_dst1)
    sp1 = cv.add(5 * laplace1, src)
    return sp1


# img = cv.imread("IMG_7583.jpg")
# img = clear_fig(img)
# sp1 = cv.imwrite("11.png", img)

# path = r'D:\yolotea6\val1'
# path_new = r'D:\yolotea6\val2'
path = r'D:\yolotea6\new_data\test'
path_new = r'D:\yolotea6\new_data\test2'
items = os.listdir(path)
for item in items:
    if item.split(".")[1] == "jpg":
        a = item.split(".")
        b = a[0] + ".png"
        patht = path + "\\" + item
        pathn = path_new + "\\" + b
        img = cv.imread(patht)
        img = clear_fig(img)
        sp1 = cv.imwrite(pathn, img)

# src = cv.imread("IMG_7583.jpg")
# src=cv.resize(src,(600,600))
# print(src.shape[:2])
# # cv.imshow("src", src)
# # 向下采样1次
# down_dst1 = cv.pyrDown(src)
# print(down_dst1.shape[:2])
# # cv.imshow("dst", down_dst1)
#
# # 向上采样1次
# up_dst1 = cv.pyrUp(down_dst1)
# print(up_dst1.shape[:2])
# # cv.imshow("up_dst1", up_dst1)
#
# # 计算拉普拉斯金字塔图像
# # 采样1次 - 向上采样1次的图
# laplace = src - up_dst1
# # cv.imshow("laplace", laplace)
# cv.waitKey()
#
# # 计算拉普拉斯金字塔图像
# # 原图 - 向上采样一次的图
# laplace1 = cv.subtract(src, up_dst1)
# # cv.imshow("laplace", laplace1)
# cv.waitKey()
#
# sp=laplace1+src
# sp1=cv.add(3*laplace1,src)
#
# res = np.hstack((src,sp1))
# # res = np.hstack((src,up_dst1,laplace,laplace1,sp,sp1))
# cv.imshow('res', res)
#
# cv.waitKey()

# import cv2 as cv
# from matplotlib import pyplot as plt
# # img = cv.imread('D:\yolotea6\\all\zi1\IMG_5863.jpg')
# img=laplace
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()
# # 6144 6160
