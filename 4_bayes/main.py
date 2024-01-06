from typing import Tuple, List, Set, Any
import numpy as np

def loadDataSet() -> Tuple[List[List[str]], List[int]]:
    """收集数据：手动创建数据集

    Returns
    -------
    postingList : List[List[str]]
        单词列表
    classVec : List[0 | 1]
        类别标签，1代表存在侮辱性言论
    """

    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], 
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

def createVocabList(dataSet: List[List[str]]) -> List[str]:
    """_summary_

    Parameters
    ----------
    dataSet : List[List[str]]
        单词列表

    Returns
    -------
    vocabList : List[str]
        统计出现过的每一个单词，汇总为列表
    """
    
    vocabSet: Set[str] = set([])
    for document in dataSet:
        # 操作符 | 用于求两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet) # vovabList

def setOfWords2Vec(vocabList: List[str], inputSet: List[str]) -> List[int]:
    """遍历查看单词是否出现，将出现该单词的位置置 1

    Parameters
    ----------
    vocabList ：List[str]
        收集的单词列表
    inputSet : List[str]
        待检索的单词列表
    
    Returns
    -------
    returnVec : List[0 | 1]
        返回带检索单词的占位信息 (单词频次表)
    """

    returnVec = [0] * len(vocabList) 
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print(f"the word: {word} is not in my Vocabulary!")
    return returnVec

def bagOfWords2Vec(vocabList, inputSet):
    """词袋模型(bag-of-words model)"""

    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def _train(
    trainMatrix: Any,
    trainCategory: Any 
) -> Tuple[List[float], List[float], float]:
    """训练数据集

    Parameters
    ----------
    trainMatrix : List[List[int]]
        文件单词矩阵
    trainCategory : List[int]
        分类标签列表

    Returns
    -------
    p0Vect : List[float]
        0 类单词在各个位置出现的概率
    p1Vect : List[float]
        1 类单词在各个位置出现的概率
    pAbusive : float
        含有侮辱性词汇（1类）词汇的样本在数据集中出现的概率
    """

    # 样本数
    numTrainDocs = len(trainMatrix)
    # 单词数
    numWords = len(trainMatrix[0])
    # 含有侮辱性词汇的样本出现的概率，即 trainCategory 中所有的 1 的个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 构造单词分类表
    p0Num = [0] * numWords
    p1Num = [0] * numWords

    # 统计整个数据集中不同分类词汇出现的概率
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        # 是否含侮辱性词汇
        if trainCategory[i] == 1:
            # 如果是，将分类表作为向量进行加和
            p1Num += trainMatrix[i]
            # 统计出现侮辱性词汇的单词总数
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    
    # 计算类别 1 下每个单词出现的概率
    p1Vect = [float(x) / p1Denom for x in p1Num]
    # 计算类别 0 下每个单词出现的概率
    p0Vect = [float(x) / p0Denom for x in p0Num]
    
    return p0Vect, p1Vect, pAbusive

def train(
    trainMatrix: Any,
    trainCategory: Any 
) -> Tuple[List[float], List[float], float]:
    # 样本数
    numTrainDocs = len(trainMatrix)
    # 单词数
    numWords = len(trainMatrix[0])
    # 含有侮辱性词汇的样本出现的概率，即 trainCategory 中所有的 1 的个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 构造单词分类表
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)

    # 统计整个数据集中不同分类词汇出现的概率
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # 是否含侮辱性词汇
        if trainCategory[i] == 1:
            # 如果是，将分类表作为向量进行加和
            p1Num += trainMatrix[i]
            # 统计出现侮辱性词汇的单词总数
            p1Denom += np.sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    
    # 计算类别 1 下每个单词出现的概率
    p1Vect = np.log(p1Num / p1Denom)
    # 计算类别 0 下每个单词出现的概率
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classify(
    vec2Classify,
    p0Vec,
    p1Vec,
    pClass1
) -> int:
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    classRet = 1 if p1 > p0 else 0
    return classRet

def testing():
    """测试朴素贝叶斯算法"""
    # 1. 加载数据集
    listPosts, listClasses = loadDataSet()
    # 2. 创建单词集合
    VocabList = createVocabList(listPosts)
    # 3. 计算单词频次表并创建数据矩阵
    trainMat = []
    for postDoc in listPosts:
        # 返回单词频次表
        trainMat.append(setOfWords2Vec(VocabList, postDoc))
    # 4. 训练数据
    trainMat = np.asarray(trainMat)
    listClasses = np.asarray(listClasses)
    p0V, p1V, pAb = _train(trainMat, listClasses)
    # 5. 测试数据
    testEntry = ["love", "my", "dalmation"]
    thisDoc = np.asarray(setOfWords2Vec(VocabList, testEntry))
    print(f"{testEntry} classified as: {classify(thisDoc, p0V, p1V, pAb)}") # ret: 0
    testEntry = ["stupid", "garbage"]
    thisDoc = np.asarray(setOfWords2Vec(VocabList, testEntry))
    print(f"{testEntry} classified as: {classify(thisDoc, p0V, p1V, pAb)}") # ret: 1

if __name__ == "__main__":
    testing()