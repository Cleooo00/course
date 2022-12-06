'''
Description: 
LastEditTime: 2022-10-17 21:27:34
FilePath: \canny\main.py
'''
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
from cannyClass import Canny
from preprocess import gray, equalization, gamma

# 灰度化
#img = gray('image\h.jpg')
img = cv2.imread('image/h_gray.jpg', 0) 
plt.figure("原始图")
plt.imshow(img, cmap='gray')
plt.show()

# 直方图均衡化
equal_img, hist, equal_hist = equalization(img)
plt.figure("原始图像直方图")
plt.plot(hist, color='b')
plt.show()

plt.figure("直方均衡化后图像直方图")
plt.plot(equal_hist, color='b')
plt.show()

plt.figure("直方图均衡化后图像")
plt.imshow(equal_img, cmap='gray')
plt.show()

# gamma矫正
gamma1_img, gamma2_img = gamma(equal_img, 1.5)
plt.figure("gamma=1.5")
plt.imshow(gamma1_img, cmap='gray')
plt.show()
plt.figure("gamma=1/1.5")
plt.imshow(gamma2_img, cmap='gray')
plt.show()


#去噪 计算梯度幅值 非极大值抑制 双阈值检测
canny = Canny()
new_gray = canny.smooth(gamma1_img)
dx, dy, M, _ = canny.gradients(new_gray)
NMS = canny.NMS(M, dx, dy)
DT = canny.double_threshold(NMS) 

plt.figure("边缘检测后图像")
plt.imshow(DT, cmap="gray")
plt.show()


# 与OpenCV结果进行对比
plt.figure(figsize=(10, 8))
plt.subplot(221)
plt.title("origin")
plt.imshow(img, cmap="gray")

plt.subplot(223)
plt.title("my canny")
plt.imshow(DT, cmap="gray")

cv_edges = cv2.Canny(img, 100, 200)
plt.subplot(224)
plt.title("OpenCV canny")
plt.imshow(cv_edges, cmap="gray")
plt.show()
