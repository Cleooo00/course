'''
Description: 包含一些图像预处理的函数
LastEditTime: 2022-10-17 21:20:37
FilePath: \canny\preprocess.py
'''



from mimetypes import init

import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

# 灰度化
def gray(img_path):
    """
    Calculate function:
    Gray(i,j) = 0.299 * R(i,j) + 0.587 * G(i,j) + 0.114 * B(i,j)
    """
    # 读取图片
    img = cv2.imread(img_path)
    # 转换成 RGB 格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 灰度化
    img_gray = np.dot(img_rgb[...,:3], [0.299, 0.587, 0.114])
    cv2.imwrite('image/h_gray.jpg', img_gray)
    return img_gray

# 直方图均衡化
def equalization(img, L=256):
    '''
    description: 
    1. 首先求出原图片的直方图 即图片中每个灰度值的具体像素点数量 具体函数为cv2.calcHist([img],[0],None,[256],[0,255])，再除以该图片的总像素点（h*w）求出其概率，并将结果放置hist数组。
    2. 利用累积分布函数，设置一个新的数组sum_hist，求出从0到i的所有灰度值所对应的像素点数的概率，即 sum_hist[i] = sum(hist[0:i+1])。
    3. 对于新建立的sum_hist，要对其乘上（L-1），并且由于灰度值是整数，所以要对结果进行四舍五入。注意此时的数组存放的键值对，是对于每个原始图片的灰度值->处理之后的图片灰度值。
    4. 最后新建图片equal_img，存放结果数据。
    return {*}
    '''     
    #img = cv2.imread(img_path, 0)# 0 灰度模式加载图片
    #cv2.imshow("ori",img)
    h, w = img.shape

    # 计算图像的直方图，即存在的每个灰度值的像素点数量
    hist = cv2.calcHist([img],[0],None,[256],[0,255])
    # 计算灰度值的像素点的概率，除以所有像素点个数，即归一化
    hist[0:255] = hist[0:255] / (h*w)
    # 设置Si
    sum_hist = np.zeros(hist.shape)
    #开始计算Si的一部分值，注意i每增大，Si都是对前i个灰度值的分布概率进行累加
    for i in range(256):
        sum_hist[i] = sum(hist[0:i+1])
    equal_hist = np.zeros(sum_hist.shape)
    #Si再乘上灰度级，再四舍五入
    for i in range(256):
        equal_hist[i] = int(((L - 1) - 0) * sum_hist[i] + 0.5)
    equal_img = img.copy()
    #新图片的创建
    for i in range(h):
        for j in range(w):
            equal_img[i, j] = equal_hist[img[i, j]]
            
    equal_hist = cv2.calcHist([equal_img], [0], None, [256], [0, 256])
    #equal_hist[0:255] = equal_hist[0:255] / (h * w)
    #cv2.imshow("inverse", equal_img)
    # 显示最初的直方图
    #plt.figure("原始图像直方图")
    #plt.plot(hist, color='b')
    #plt.show()
    #plt.figure("直方均衡化后图像直方图")
    #plt.plot(equal_hist, color='b')
    #plt.show()
    #cv2.waitKey()
    #return equal_hist
    return equal_img, hist, equal_hist

# gamma 矫正
def gamma(img, r=1.5):   
    #img = cv2.imread(img_path, 0)
    img1 = np.power(img/float(np.max(img)), 1/r)
    img2 = np.power(img/float(np.max(img)), r)
    #imgs = np.hstack((img, img1, img2))
    #cv2.imshow('org, gamma=1.5, gamma=1/1.5',imgs)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return img1,img2
