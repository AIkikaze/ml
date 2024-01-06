import main as bayes
import numpy as np

def textParse(bigString):
    """接收一个文段，将其解析为单词列表"""

    import re
    # 使用正则表达式来切分句子，其中分隔符为除单词、数字外的任意字符串
    listOfWords = re.findall(r'\b\w+\b', bigString.lower())
    return [word for word in listOfWords if len(word) > 2]

def calMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    # 遍历词汇表中的每个词
    for token in vocabList: 
        # 统计单词出现的次数
        freqDict[token] = fullText.count(token)
    # 根据每个单词出现的次数由高到底排序
    sortedFreq = sorted(freqDict.items(), key=lambda x: x[1], reverse=True)
    # 返回出现次数最高的 30 个单词
    return sortedFreq[:30]

def localWords(feed1, feed0):
    docList = [] 
    classList = []
    fullText = []
    minLen = min(len(feed1["entries"]), len(feed0["entries"]))

    # 1. 文本获取与统计
    for i in range(minLen):
        # 类别 1：每次访问一条 RSS 源
        wordList = textParse(feed1["entries"][i]["summary"])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # 类别 0：每次访问一条 RSS 源
        wordList = textParse(feed0["entries"][i]["summary"])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = bayes.createVocabList(docList)
    top30Words = calMostFreq(vocabList, fullText)

    # 2. 文本处理
    # 去掉出现次数最高的前 30 个单词（基本都是冠词、连词、系动词、情态动词这种
    # 没有太多实际含义的词
    for wordCount in top30Words:
        if wordCount[0] in vocabList:
            vocabList.remove(wordCount[0])
    
    # 3. 训练-验证集划分
    trainSet = list(range(2 * minLen))
    testSet = []
    for i in range(5):
        randIndex = np.random.choice(len(trainSet))
        testSet.append(trainSet[randIndex])
        trainSet.pop(randIndex)
    trainMat = []
    trainClasses = []
    for docIndex in trainSet:
        trainMat.append(bayes.bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    
    # 4. 朴素贝叶斯算法
    trainMat = np.asarray(trainMat)
    trainClasses = np.asarray(trainClasses)
    p0V, p1V, pClass1 = bayes.train(trainMat, trainClasses)

    # 5. 算法验证
    errorCount = 0
    for docIndex in testSet:
        wordVec = np.asarray(bayes.bagOfWords2Vec(vocabList, docList[docIndex]))
        if bayes.classify(wordVec, p0V, p1V, pClass1) != classList[docIndex]:
            errorCount += 1
    print(f"the error rate is: {100. * float(errorCount) / len(testSet)} %")
    return vocabList, p0V, p1V

def getTopWords(feed1, feed0):
    vocabList, p0V, p1V = localWords(feed1, feed0)
    top1Words = []
    top0Words = []
    for i in range(len(p0V)):
        if p0V[i] > -5.0: # 为什么这里是 -6.0 呢？
            top0Words.append((vocabList[i], p0V[i]))
        if p1V[i] > -5.0:
            top1Words.append((vocabList[i], p1V[i]))

    sortedWords = sorted(top0Words, key=lambda x: x[1], reverse=True)
    print("----- Class 0 Top Words ------")
    for item in sortedWords: print(item[0])

    sortedWords = sorted(top1Words, key=lambda x: x[1], reverse=True)
    print("----- Class 1 Top Words ------")
    for item in sortedWords: print(item[0])

if __name__ == "__main__":
    import feedparser as fp # type: ignore
    fd0 = fp.parse('https://www.lesswrong.com/feed.xml?view=curated-rss')
    fd1 = fp.parse('https://www.knowablemagazine.org/rss')
    localWords(fd1, fd0)
    # getTopWords(fd1, fd0)
