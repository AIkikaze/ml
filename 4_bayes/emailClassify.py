import main as bayes
import numpy as np

def textParse(bigString):
    """接收一个文段，将其解析为单词列表"""

    import re
    # 使用正则表达式来切分句子，其中分隔符为除单词、数字外的任意字符串
    listOfWords = re.findall(r'\b\w+\b', bigString.lower())
    return [word for word in listOfWords if len(word) > 2]

def spamTest():
    """使用朴素贝叶斯算法进行垃圾邮件分类"""

    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # 读取文本：解析数据并归类
        with open(f"4_bayes/email/spam/{i}.txt", "r", encoding='latin-1') as f:
            email = f.read()
        wordList = textParse(email)
        docList.append(wordList)
        classList.append(1)
        # 读取文本：解析数据并归类
        with open(f"4_bayes/email/ham/{i}.txt", "r", encoding='latin-1') as f:
            email = f.read()
        wordList = textParse(email)
        docList.append(wordList)
        classList.append(0)
    # 创建词汇表
    vocabList = bayes.createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    # 随机取 10 个邮件进行测试
    for i in range(10):
        randIndex = np.random.choice(len(trainingSet))
        testSet.append(trainingSet[randIndex])
        trainingSet.pop(randIndex)
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        sampleVec = bayes.setOfWords2Vec(vocabList, docList[docIndex])
        trainMat.append(sampleVec)
        trainClasses.append(classList[docIndex])
    # 训练数据
    trainMat = np.asarray(trainMat)
    trainClasses = np.asarray(trainClasses)
    p0V, p1V, pSpam = bayes.train(trainMat, trainClasses)

    # 验证拟合结果
    errorCount = 0
    for docIndex in testSet:
        wordVec = np.asarray(bayes.setOfWords2Vec(vocabList, docList[docIndex]))
        if bayes.classify(wordVec, p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    
    print(f"the errorCount is: {errorCount}")
    print(f"the text length is: {len(testSet)}")
    print(f"the error rate is: {100. * float(errorCount) / len(testSet)} %")

if __name__ == "__main__":
    spamTest()