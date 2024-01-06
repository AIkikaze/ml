import numpy as np
import matplotlib.pyplot as plt # type: ignore

def loadDataSet(file_name):
    """加载并且解析数据"""

    dataList = []
    labelList = []
    with open(file_name, 'r') as file:
        for line in file.readlines():
            lineArr = [float(x) for x in line.strip().split('\t')]
            dataList.append(lineArr[:-1])
            labelList.append(lineArr[-1])
    return dataList, labelList 

# h_w(x) = g(w'x) = 1 / (1 + e^{-w'x})
def sigmoid(inX):
    return 1.0 / (1.0 + np.exp(-inX))

def gradAscent(dataMat, labelVec):
    # 获取特征个数和样本数
    numSample, numFeature = dataMat.shape[:2]
    # 设置迭代步长
    alpha = 0.001
    # 最大迭代次数
    maxIterations = 500
    # 特征参数向量(列向量)
    weights = np.ones((numFeature, 1))

    """
    参考: https://blog.csdn.net/achuo/article/details/51160101

    目标函数
    J(x) = -1/m * \sum_{i=1}^m ( (y^i log h_w(x^i) + (1-y^i) log (1 - h_w(x^i)))
    注: x 输入的随机变量, y 输出的分类 0/1, i 向量下标, h_w(x) 等价于这里的 sigmoid(weights @ x)

    求导
    >> \partial_j J(x) = 1/m * \sum_{i=1}^m (h_w(x^i) - y^i) x_j^i
    >> \partial J(x) = 1/m * x'(h_w(x) - y)

    故有迭代公式
    w_{n+1} = w_n - alpha * \partial J(x) = w_n - alpha/m * x'(h_w(x) - y)
    """
 
    # 设置初值为 0
    iters = 1 
    # 开始迭代
    while iters < maxIterations:
        # (numSamples, numFeatures) x (numFeatures, 1) -> (numSamples, 1)
        classRet = sigmoid(dataMat @ weights)
        error = classRet - labelVec
        weights = weights - alpha * dataMat.T @ error
        iters += 1
    
    return weights

def stocGradAscent0(dataMat, labelVec):
    numSamples, numFeature = dataMat.shape[:2]
    alpha = 0.01
    weights = np.ones(numFeature)
    maxIterations = numSamples

    iters = 0
    while iters < maxIterations:
        id = iters
        # 每次取一个样本来计算梯度
        classRet = sigmoid(np.dot(dataMat[id], weights))
        error = classRet - labelVec[id]
        weights = weights - alpha * error * dataMat[id]
        iters += 1
    
    return weights

def plotBestFit(dataMat, labelVec, weights):
    labels = np.unique(labelVec)
    labelList = labelVec.flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    X = dataMat[labelList == labels[0], :]
    ax.scatter(X[:, 1], X[:, 2], s=30, color='red', marker='s')

    X = dataMat[labelList == labels[1], :]
    ax.scatter(X[:, 1], X[:, 2], s=30, color='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x) / weights[2]
    ax.plot(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

if __name__ == "__main__":
    dataList, labelList = loadDataSet("5_logistic/TestSet.txt")
    # 数据格式处理
    # 1. dataMat = [[1, x1, x2, ..., xn], ...] 以便与 w 特征参数作内积
    # 2. labelMat 转为 numpy 列向量
    dataMat = np.asarray(dataList)
    dataMat = np.hstack([np.ones((dataMat.shape[0], 1)), dataMat])
    labelVec = np.asarray([labelList]).T

    # 计算特征参数
    # weights = gradAscent(dataMat, labelVec)
    weights = stocGradAscent0(dataMat, labelVec)

    # 数据可视化
    plotBestFit(dataMat, labelVec, weights)
    
