import os
import numpy as np
from tqdm import tqdm

def load(file_path):
    vec = np.zeros((32, 32))
    fr = open(file_path)
    for i in range(32):
        # 按行读取
        line_str = fr.readline()
        # str 最后以 '\n' 结尾
        numbers_list = [int(char) for char in line_str[:-1]]
        # 转为 numpy 数组
        vec[i, :] = np.asarray(numbers_list[:32])
    
    vec = np.ravel(vec)
    return vec

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

def test():
    # 导入训练数据
    file_list = os.listdir("2_digits/trainingDigits") 
    testing_size = len(file_list)
    traing_labels = np.zeros((testing_size,))
    traing_mat = np.zeros((testing_size, 1024))
    for i, file_str in enumerate(file_list):
        # 去除文件后缀 .txt
        file_name = file_str.split('.')[0]
        # 从文件名称读取数字标签
        traing_labels[i] = int(file_name.split('_')[0])
        # 读取文本数据
        traing_mat[i, :] = load(f"2_digits/trainingDigits/{file_str}")
    
    # 导入测试数据
    file_list = os.listdir("2_digits/testDigits") 
    testing_size = len(file_list)
    error_count = 0.0
    for i, file_str in tqdm(enumerate(file_list), desc="Test", unit="files"):
        # 去除文件后缀 .txt
        file_name = file_str.split('.')[0]
        # 从文件名称读取数字标签
        number = int(file_name.split('_')[0])
        # 读取文本数据
        inst = load(f"2_digits/testDigits/{file_str}")
        # knn 分类
        ret = classify_knn(inst, traing_mat, traing_labels, 3)
        # 结果对比
        if ret != number: error_count += 1.0
    
    print(f":: 测试样本数量={testing_size} ")
    print(f":: 错误率: {error_count} / {testing_size} = {error_count / testing_size}")

if __name__ == "__main__":
    test()