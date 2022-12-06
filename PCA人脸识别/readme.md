#### 1.环境 

python 3.9.7

#### 2.输入输出

利用训练集 $TrainDataBase$二十个数据进行训练，抽取人脸主成分，构成特征空间。识别时将测试集$TestDataBase$十个数据投影到此空间，通过与人脸图像比对并输出结果。

#### 3.结果分析

**原始的$PCA$方法**

1. 取原始数据，构成矩阵X

2. 将X每一行进行0均值化，得到矩阵P

   ```python
   mx = X.mean(axis=1)
   P = X - mx
   ```

3. 求出P的协方差矩阵$Cov\_P = \frac{1}{m}PP^T$

4. 求出协方差矩阵对应的特征值以及对应的特征向量

   ```python
   D, V = np.linalg.eig(Cov_P)
   ```

5. 将特征向量按照对应的特征值大小从上到下按行构成矩阵，取前k行组成矩阵，本例中，k取20。

   ```python
   V1 = V[:, 0:PCA_NUM]  # 取前20个特征向量
   V1 = P.T * V1
   for i in range(PCA_NUM):  # 特征向量归一化
       L = np.linalg.norm(V1[:, i])
       V1[:, i] = V1[:, i] / L
   ```

6. 将P与构成特征矩阵相乘得到降维后数据

**识别过程**

对测试集也做相似的过程，构造矩阵、均值化处理，然后利用前面$PCA$方法求得的特征矩阵左乘测试集矩阵，对其进行空间变换。然后依次将训练集中数据拿过来做对比，找到最相近的图片（向量距离最小）

**验证过程**

构造一个测试集和训练集的对应数组，每当识别正确一次，则记录，由此计算正确率。

```python
ACCURACY_DATA = [[1, 2], [7, 8], [3, 4], [9, 10], [13, 14], [5, 6], [17, 18], [15, 16], [19, 20], [11, 12]]
def accuracy(test_data_index, recogize_img_index):
    return recogize_img_index in ACCURACY_DATA[test_data_index-1]
```

最终十张图片均识别正确

```python
accuracy percent: 100.00%
```

