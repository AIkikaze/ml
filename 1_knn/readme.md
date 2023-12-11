## 项目案例1: 优化约会网站的配对效果

### 项目概述

海伦使用约会网站寻找约会对象。经过一段时间之后，她对约会对象会有三种不同的感觉: `根本不喜欢`, `有点喜欢`, `应该很喜欢`。

现在她收集到了一些约会网站未曾记录的数据信息，这更有助于匹配对象的分类。

### 开发流程

收集数据: 提供文本文件

准备数据: 使用 Python 解析文本文件

分析数据: 使用 Matplotlib 画二维散点图

训练算法: 此步骤不适用于 k-近邻算法

测试算法: 使用海伦提供的部分数据作为测试样本。测试样本和非测试样本的区别在于: 测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。

使用算法: 产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型。

#### 收集数据

文本文件数据格式如下:

```json
40920	8.326976	0.953952	3
14488	7.153469	1.673904	2
26052	1.441871	0.805124	1
75136	13.147394	0.428964	1
38344	1.669788	0.134296	1
...
```

#### 收集数据

将文本解析为 numpy 数组类型

```python
def file2matrix(file_path):
    """_summary_
        导入数据集
    Args:
        filepath (string): 数据文件路径 
    Returns:
        mat, labels:
        数据矩阵和对应的类别标签
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
```

### 分析数据

使用 Matplotlib 画二维散点图, 由图可以看出三种类别的人各自的大致分布

![Matplotlib 散点图](/1_knn/view.png)

之后, 归一化来消除特征之间量级不同导致的影响

```python
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
```

### 训练算法

这个过程在这里没有特别的必要，因为不涉及太多的参数调整。这里我们根据 knn 算法的原理, 用代码实现就可以了。

```python
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
```

### 测试算法

取测试数据的一部分来验证算法的结果，给出错误率

```python
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
```

### 使用算法

对一个新的约会对象进行分类预测

```python
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
```
 
