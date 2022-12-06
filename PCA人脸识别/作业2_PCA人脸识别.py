import numpy as np
import cv2 as cv
import os

IMAGE_SIZE = (50, 50)
PCA_NUM = 20
ACCURACY_DATA = [[1, 2], [7, 8], [3, 4], [9, 10], [13, 14], [5, 6], [17, 18], [15, 16], [19, 20], [11, 12]]


# 1、加载训练集中的脸，转为一个M行N列矩阵T
def load_data(path):
    # 查看路径下所有文件
    train_file_path = os.listdir(path)
    # 计算有几个文件（图片命名都是以 序号.jpg方式）减去Thumbs.db
    file_number = len(train_file_path) - 1
    T = []
    # 把所有图片转为1-D并存入T中
    for i in range(1, file_number + 1):
        image = cv.imread(path + '/' + str(i) + '.jpg', cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, IMAGE_SIZE)
        # 转为1-D
        image = image.reshape(image.size, 1)
        T.append(image)
    T = np.array(T)
    # 不能直接T.reshape(T.shape[1],T.shape[0]) 这样会打乱顺序，
    T = T.reshape(T.shape[0], T.shape[1])
    return np.mat(T).T


def PCA(X):
    #0均值化
    mx = X.mean(axis=1)
    P = X - mx
    temp = P * P.T
    D, V = np.linalg.eig(temp)  # 特征值与特征向量
    V1 = V[:, 0:PCA_NUM]  # 取前20个特征向量
    V1 = P.T * V1
    for i in range(PCA_NUM):  # 特征向量归一化
        L = np.linalg.norm(V1[:, i])
        V1[:, i] = V1[:, i] / L
    U = P * V1  # 降维后的数据
    return U, mx, P


def recognize(testImagePath, U, mx, P):
    # 取特征矩阵列数，作为训练次数
    _, trainNumber = np.shape(U)
    # projectImage为训练集图片，P存储整个训练集图片矩阵，U为其协方差矩阵的特征矩阵
    projectImage = U.T * (P)
    # 测试图片的读取 imdecode可以看出哦imread此处用来处理中文字符，cv2.IMREAD_GRAYSCALE灰度图片读取
    testImageArray = cv.imdecode(np.fromfile(testImagePath, dtype=np.uint8), cv.IMREAD_GRAYSCALE)
    # 大小设置保持与测试集一致
    testImageArray = cv.resize(testImageArray, IMAGE_SIZE)
    # 将其转化为1-D的形式，用于和projectImage中的 图片数量-D中的挨个进行比较距离大小
    testImageArray = testImageArray.reshape(testImageArray.size, 1)
    # 转化为矩阵形式
    testImageArray = np.mat(testImageArray)
    # 通过传入的mx,U对测试集图像进行同样的处理
    # 均值化处理
    differenceTestImage = testImageArray - mx
    # k-l变换的到处理后的测试图像
    projectTestImage = U.T * differenceTestImage
    # 距离记录
    distance = []
    for i in range(0, trainNumber):
        # 挨个取出每个测试集图片参数q
        q = projectImage[:, i]
        # 比较训练集和测试集距离大小
        temp = np.linalg.norm(projectTestImage - q)
        # 记录距离
        distance.append(temp)
    # 取得最小距离
    minDistance = min(distance)
    # 取得最小距离对应的index
    index = distance.index(minDistance)
    # 展示测试集数据图像
    img_test = cv.imread(testImagePath, cv.IMREAD_GRAYSCALE)
    # 展示输出集合测试图像
    img_recognize = cv.imread('./TrainDatabase' + '/' + str(index + 1) + '.jpg', cv.IMREAD_GRAYSCALE)
    cv.imshow('test data and recognize result',  np.hstack((img_test, img_recognize)))
    # 循环记录为从0开始，图片命名从1开始，故此处为index+1
    return index + 1


# 判定正确率，每识别正确一次，返回1
def accuracy(test_data_index, recogize_img_index):
    return recogize_img_index in ACCURACY_DATA[test_data_index-1]


if __name__ == "__main__":
    X = load_data('TrainDatabase')
    # kl转换得到相关参数
    U, mx, P = PCA(X)
    accuracy_num = 0
    for i in range(1, 11):
        testImagePath = './TestDatabase' + '/' + str(i) + '.jpg'
        recogize_img_index = recognize(testImagePath, U, mx, P)
        accuracy_num += accuracy(i, recogize_img_index)
        cv.waitKey(0)
    print('accuracy percent: {:.2%}'.format(accuracy_num / 10))