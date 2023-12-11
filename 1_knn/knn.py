import os
import numpy as np
from tqdm import tqdm

def file2matrix(file_path):
    """_summary_
        导入数据集
    Args:
        filepath (string): 数据文件路径 
    Returns:
        数据矩阵 returnMat 和对应的类别 classLabelVector
    """
    fr = open(file_path)
    data = fr.readlines()
    number_of_lines = len(data)
    mat = np.zeros((number_of_lines, 3))
    labels = []

    for i, line in enumerate(data):
        # 去除开头和结尾的空白字符
        line = line.strip() 
        # 读取每列的数据
        list_from_line = line.split('\t')
        # 存入矩阵
        mat[i, :] = list_from_line[:3]
        # 加入 label 标签
        labels.append(int(list_from_line[-1]))
    
    return mat, np.asarray(labels)

def auto_norm(data_set):
    """_summary_
        数据特征归一化, 消除特征数量级不同导致的影响
    Args:
        data_set (np.array): 数据集 
    Returns:
        归一化公式
        normalized = (x - xmin) / (xmax - xmin)
    """
    _, dimension = data_set.shape[:2]
    normalized = np.zeros_like(data_set)

    for i in range(dimension):
        vmin = np.min(data_set[:, i])
        vmax = np.max(data_set[:, i])
        normalized[:, i] = (data_set[:, i] - vmin) / (vmax - vmin)
    
    return normalized

def classify_knn(inst, data_set, labels, k):
    """_summary_
        用 k-近邻算法 实现的分类模型
    Args:
        inst (np.array): 被分类的个例
        data_set (np.array): 数据集
        labels (np.array): 标签
        k (int): 选择前 k 个距离目标个例最近的数据进行类别投票
    Returns:
        根据投票数量的多少, 确定预测的结果
    """
    # 计算实例 inst 到 data_set 的相对距离
    dists = np.linalg.norm(data_set - inst, axis=1)
    # 取距离最小的前 n 个数据, 将其标签存到 used_labels 中
    indices = np.argsort(dists)[:k]
    used_labels = labels[indices]
    # 使用 np.unique 获取唯一元素和对应的出现次数
    unique_labels, label_counts = np.unique(used_labels, return_counts=True)
    # 找到出现次数最多的标签
    most_common_label = unique_labels[np.argmax(label_counts)]

    return most_common_label

def View():
    datingDataMat, datingLabels = file2matrix("1_knn/datingTestSet.txt")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
    plt.show()

def Test(ratio=0.2, k=5):
    dating_data_set, labels = file2matrix("1_knn/datingTestSet.txt")
    normalized = auto_norm(dating_data_set)
    size = normalized.shape[0]
    size_of_test = int(size * ratio)
    error_count = 0.0

    for i in tqdm(range(size_of_test)):
        ret = classify_knn(normalized[i, :], normalized[size_of_test:], labels[size_of_test:], k)
        # print(f">> 分类结果: {ret} - {labels[i]}")
        if ret != labels[i]:
            error_count += 1.0

    print(f":: 测试样本数量={size_of_test} ")
    print(f":: 错误率: {error_count} / {size_of_test} = {error_count / size_of_test}")

def classify_person():
    likeability = ['根本不', '有点儿', '应该很']
    ice_cream = float(input("每周消费的冰淇淋公升数?"))
    fly_miles = int(input("每年获得的飞行常客里程数?"))
    game_time = float(input("玩视频游戏所耗时间百分比?"))
    
    dating_data_set, labels = file2matrix("1_knn/datingTestSet.txt")
    normalized = auto_norm(dating_data_set)
    inst = np.array([fly_miles, ice_cream, game_time])
    idx = classify_knn(inst, normalized, labels, 5)
    print(f'>> 推测你 "{likeability[idx]}" 喜欢这个人.')
 

if __name__ == "__main__":
    View()
    Test()
    classify_person()