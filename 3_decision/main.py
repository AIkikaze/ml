import math
from pprint import pprint

def createDataSet():
    dataSet = [
        [1, 1, "yes"],
        [1, 1, "yes"],
        [1, 0, "no"],
        [0, 1, "no"],
        [0, 1, "no"]
    ]
    labels = ["no surfacing", "flippers"]
    return dataSet, labels

def calShannonEnt(dataSet, debug=False):
    # 求 list 长度，表示参数训练的数据量
    numEntries = len(dataSet)
    # 计算分类标签 label 出现的次数
    labelCounts = {}
    # 统计所有可能的分类并为其计数
    for featVec in dataSet:
        # 储存当前的标签，子列中最后一个元素
        currentLabel = featVec[-1]
        # 为所有可能的分类创建字典，如果当前键值不存在，则扩展字典并将当前
        # 键值加入字典。每个键值都记录了当前类别出现的次数。
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    if debug:
        print(f">> labelCounts: {labelCounts}")
        # >> labelCounts: {'yes': 2, 'no': 3}
    
    # 通过计算 label 标签的占比，求出香农熵
    shannonEnt = 0.0
    for key in labelCounts.keys():
        # 使用所有类标签的发生频率计算类别出现的概率
        prob = float(labelCounts[key]) / numEntries
        # 计算香农熵，以 2 为底求对数
        shannonEnt -= prob * math.log(prob, 2)
    # >> shanonEnt = -[ 2/5 * log(2/5, 2) + 3/5 * log(3/5, 2) ]
    return shannonEnt

def splitDataSet(dataSet, index, value, debug=False):
    """按照指定的特征划分数据集

    Parameters
    ----------
    dataSet : List[Any]
        数据集 - 待划分的数据集
    index : int
        列指标 - 表示某个特征
    value : int
        特征约束 - 表示特征的值约束

    Returns
    -------
    List[Any]
        划分好的数据集
    """
    retDataSet = []
    for featVec in dataSet:
        # index 列为 value 的数据集 [该数据集需要排除 index 列]
        if featVec[index] == value:
            # 取子列的前 index 个元素
            reduceFeatVec = featVec[:index]
            # 与剩余的元素分割开来
            reduceFeatVec.extend(featVec[index+1:])
            # 收集结果值
            retDataSet.append(reduceFeatVec)
    if debug:
        print(f">> retDataSet: {retDataSet}")
    return retDataSet

def chooseBestFeatureToSplit(dataSet, debug=False):
    """选择最佳特征

    Args:
        dataSet 数据集
    Returns:
        bestFeature 最优特征列
    """
    # 求一行有多少列的 Feature，最后一列是 label 列
    numFeatures = len(dataSet[0]) - 1
    # 数据集的原始信息熵
    baseEntropy = calShannonEnt(dataSet)
    # 最优的信息增益值和最优的 Feature 编号
    bestInfoGain, bestFeature = 0.0, -1
    # 遍历所有特征
    for i in range(numFeatures):
        # 获取对应 Feature 下的所有数据
        featList = [example[i] for example in dataSet]
        # 获取剔重后的集合，使用 set 对 list 进行去重
        uniqueVals = set(featList)
        # 创建一个临时的信息熵
        newEntropy = 0.0
        # 遍历某一列的 value 结合，计算该列的信息熵
        # 遍历当前特征中所有唯一属性值，对每个属性值划分一次数据集
        # 计算数据集的新熵值，并对所有唯一特征值得到的熵求和
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算概率
            prob = len(subDataSet) / float(len(dataSet))
            # 计算信息熵
            newEntropy += prob * calShannonEnt(subDataSet)
        # gain[信息增益]: 划分数据集前后的信息编号，划去信息熵的最大的值
        # 信息增益是熵的减少（数据无序程度的减少）。最后，比较所有特征中的
        # 信息增益，返回最好特征划分的索引值。
        infoGain = baseEntropy - newEntropy

        if debug:
            print(f"--- Feature [{i}] ---\n"
                  +f": Entropy: {baseEntropy} -> {newEntropy}\n"
                  +f"> infoGain= {infoGain}")

        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    if debug:
        print(f">> Best Feature is [{bestFeature}]")
    return bestFeature

def majorityCnt(classList, debug=False):
    """选择出现次数最多的一个结果

    Parameters
    ----------
    classList : _type_
        标签列的集合
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)

    if debug:
        print(list(classCount.items()))
        print("--- sorted ---\n"+f"{sortedClassCount}")

    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 如果数据集最后一列的第一个值出现的次数 = 整个集合的数量
    # 也就是说只有一个类别，就直接返回结果就行
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 如果数据集只有一列，那么以出现特征次数最多的特征作为结果
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    
    # 选择最优特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取标签名称
    bestFeatLabel = labels[bestFeat]
    # 初始化建树
    Tree = {bestFeatLabel: {} }
    # 取出最优列，然后对它的分支进行分类
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签
        subLabels = list(filter(lambda x: x != bestFeatLabel, labels))
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归建树
        Tree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return Tree

def classify(inputTree, featLabels, testVec, debug=False):
    """输入决策树节点，进行分类

    Parameters
    ----------
    inputTree : _type_
        决策树模型
    featLabels : _type_
        Feature 标签对应的名称
    testVec : _type_
        测试输入的数据
    
    Returns
    ---------
        classLabels 分类的结果
    """
    # 获取 tree 根节点对应的 key 值
    firstStr = list(inputTree.keys())[0]
    # 通过 key 得到根节点对应的 value
    secondDict = inputTree[firstStr]
    # 通过根节点名称获取根节点在 Label 中的先后顺序，从而进行决策
    featIndex = featLabels.index(firstStr)
    # 测试数据，找到根节点对应的 Label 位置，也就知道从输入数据的第几位开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]

    if debug:
        print(f"--- local tree of [{firstStr}]---")
        pprint(secondDict, depth=2, compact=True)
        print(f"--- judging process on <{key}> ---")
        pprint(valueOfFeat, depth=2, compact=True)
    
    # 判断分支是否结束
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec, debug)
    else:
        classLabel = valueOfFeat
    return classLabel

vec = [False] * 100
def print_tree(node, indent=0):
    kids = list(node.keys())

    while True:
        if indent > 0:
            for i in range(indent-1):
                print("│   " if vec[i] else "    ", end="")
            print("├── " if vec[indent-1] else "└── ", end="")

        head_str = kids[0]
        if not isinstance(node[head_str], dict):
            print(f"{head_str} -> {node[head_str]}", end="\n")
            return
        else:
            print(head_str, end="\n")
            node = node[head_str]
            kids = list(node.keys())

        if len(kids) > 1: break
        indent += 1

    vec[indent] = True
    print_tree({kids[0] : node[kids[0]]}, indent+1)
    vec[indent] = False
    print_tree({kids[1] : node[kids[1]]}, indent+1)

def fishTest():
    """对动物是否为鱼类分类的测试函数"""

    # 1. 创建数据和结果标签
    dataSet, labels = createDataSet()
    # 2. 构建决策树
    tree = createTree(dataSet, labels)
    # 3. 进行预测
    inst = [0, 0]
    inst[0] = int(input(f"Is it {labels[0]}? [1/0]\n>> "))
    inst[1] = int(input(f"Has {labels[1]}? [1/0]\n>> "))
    print(f">> Is a fish? {classify(tree, labels, inst, debug=True)}.")
    # 4. 输出决策树
    print("----- Tree View -----")
    print_tree(tree)

def contactLensesTest():
    """预测隐形眼镜的测试函数"""
    import pandas as pd # type: ignore

    # 1. 加载数据集和标签
    labels = ["age", "prescript", "astigmatic", "tearRate", "target"]
    df = pd.read_csv("3_decision/lenses.txt", sep='\t', names=labels)
    dataSet = df.values.tolist()
    # 2. 构建决策树
    tree = createTree(dataSet, labels)
    # 3. 进行预测
    testVec = dataSet[:5]
    y_predict = [classify(tree, labels, vec) for vec in testVec]
    y = [example[4] for example in dataSet[:5]] 
    # 4. 输出预测结果
    print(f">> predict: {y_predict}\n>> answer:  {y}")
    # 5. 输出决策树
    print("----- Tree View -----")
    print_tree(tree)

if __name__ == "__main__":
    contactLensesTest()