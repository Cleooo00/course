import cv2
import numpy as np
import math


def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gauss_kernel(size, sigma):
    gauss = np.zeros((size, size), np.float32)
    center = size//2
    for i in range(size):
        for j in range(size):
            numerator = math.pow(i-center, 2) + pow(j-center, 2)  # 当size=3时，中心点 （x0，y0） （1，1）
            gauss[i, j] = math.exp(-numerator/(2*math.pow(sigma, 2)))/(math.pow(sigma, 2)*2*math.pi)
    sum = np.sum(gauss)
    kernel = gauss/sum
    return kernel


def gauss_filter(img, kernel):
    # bgr三通道
    if len(img.shape) == 3:
        h, w, c = img.shape
        img1 = np.zeros((h, w, c), np.uint8)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                for c1 in range(c):
                    sum = 0
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            sum += img[i + k, j + l, c1] * kernel[k + 1, l + 1]
                    img1[i, j, c1] = sum
    # 灰度
    else:
        h = img.shape[0]
        w = img.shape[1]
        img1 = np.zeros((h, w), np.uint8)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                sum = 0
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        sum += img[i + k, j + l] * kernel[k + 1, l + 1]
                img1[i, j] = sum
    return img1


# 手写生成核函数
size, sigma = 3, 1.5
kernel = gauss_kernel(size, sigma)
print(kernel)

# 调用库函数生成核函数
kernel_1d = cv2.getGaussianKernel(size, sigma)
kernel_2d = kernel_1d * kernel_1d.T  # 向量*转置得到二维矩阵
print(kernel_2d)


# 输入彩色图像，进行滤波处理
img = cv2.imread('image\lane.jpg')
print('原始图像 shape: ', img.shape)
filter_img = gauss_filter(img, kernel)
show('filter_img', filter_img)
print(filter_img)

# 将图片灰度处理后，进行滤波
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print('灰度处理后，shape：', img_gray.shape)
show('input_gray_img', img_gray)
filter_img_gray = gauss_filter(img_gray, kernel)
show('filter_img_gray', filter_img_gray)

# 直接调用库函数进行滤波
out = cv2.GaussianBlur(img, (size, size), sigma)
show('api_gauss_filter_img', out)
print(out)






