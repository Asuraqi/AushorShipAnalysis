import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier

def KNN():
    clf = neighbors.KNeighborsClassifier()
    return clf

#线性鉴别分析（Linear Discriminant Analysis）
def LDA():
    clf = LinearDiscriminantAnalysis()
    return clf

#支持向量机（Support Vector Machine）
def SVM():
    clf = svm.SVC()
    return clf

#逻辑回归（Logistic Regression）
def LR():
    clf = LogisticRegression()
    return clf

#随机森林决策树（Random Forest）
def RF():
    clf = RandomForestClassifier()
    return clf

#多项式朴素贝叶斯分类器
def native_bayes_classifier():
    clf = MultinomialNB(alpha = 0.01)
    return clf

#决策树
def decision_tree_classifier():
    clf = tree.DecisionTreeClassifier()
    return clf

#GBDT
def gradient_boosting_classifier():
    clf = GradientBoostingClassifier(n_estimators = 200)
    return clf
#计算识别率

# def report(results, n_top=5488):
#     f = open('E:\grid_search_rf.txt', 'w')
#     for i in range(1, n_top + 1):
#         candidates = np.flatnonzero(results['rank_test_score'] == i)
#         for candidate in candidates:
#             f.write("Model with rank: {0}".format(i) + '\n')
#             f.write("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#                   results['mean_test_score'][candidate],
#                   results['std_test_score'][candidate]) + '\n')
#             f.write("Parameters: {0}".format(results['params'][candidate]) + '\n')
#             f.write("\n")
#     f.close()

#自动调参（以随机森林为例）
def selectRFParam():
    clf_RF = RF()
    param_RF = {"max_depth": [3,15],
                  "min_samples_split": [3, 5, 10],
                  "min_samples_leaf": [3, 5, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"],
                  "n_estimators": range(10,50,10)}
                  # "class_weight": [{0:1,1:13.24503311,2:1.315789474,3:12.42236025,4:8.163265306,5:31.25,6:4.77326969,7:19.41747573}],
                  # "max_features": range(3,10),
                  # "warm_start": [True, False],
                  # "oob_score": [True, False],
                  # "verbose": [True, False]}
    grid_search = GridSearchCV(clf_RF, param_grid=param_RF, n_jobs=4)
    start = time()
    T = getData_2()    #获取数据集
    grid_search.fit(T[0], T[1]) #传入训练集矩阵和训练样本类标
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)

def totalAlgorithm_1():
    #获取各个分类器
    clf_KNN = KNN()
    clf_LDA = LDA()
    clf_SVM = SVM()
    clf_LR = LR()
    clf_RF = RF()
    clf_NBC = native_bayes_classifier()
    clf_DTC = decision_tree_classifier()
    clf_GBDT = gradient_boosting_classifier()
    #获取训练集和测试集
    setDict = getData_3()
    setNums = len(setDict.keys()) / 4  #一共生成了setNums个训练集和setNums个测试集，它们之间是一一对应关系
    #定义变量，用于将每个分类器的所有识别率累加
    KNN_rate = 0.0
    LDA_rate = 0.0
    SVM_rate = 0.0
    LR_rate = 0.0
    RF_rate = 0.0
    NBC_rate = 0.0
    DTC_rate = 0.0
    GBDT_rate = 0.0
    for i in range(1, int(setNums + 1)):
        trainMatrix = setDict[str(i) + 'train']
        trainClass = setDict[str(i) + 'trainclass']
        testMatrix = setDict[str(i) + 'test']
        testClass = setDict[str(i) + 'testclass']
        #输入训练样本
        clf_KNN.fit(trainMatrix, trainClass)
        clf_LDA.fit(trainMatrix, trainClass)
        clf_SVM.fit(trainMatrix, trainClass)
        clf_LR.fit(trainMatrix, trainClass)
        clf_RF.fit(trainMatrix, trainClass)
        clf_NBC.fit(trainMatrix, trainClass)
        clf_DTC.fit(trainMatrix, trainClass)
        clf_GBDT.fit(trainMatrix, trainClass)
        #计算识别率
        KNN_rate += getRecognitionRate(clf_KNN.predict(testMatrix), testClass)
        LDA_rate += getRecognitionRate(clf_LDA.predict(testMatrix), testClass)
        SVM_rate += getRecognitionRate(clf_SVM.predict(testMatrix), testClass)
        LR_rate += getRecognitionRate(clf_LR.predict(testMatrix), testClass)
        RF_rate += getRecognitionRate(clf_RF.predict(testMatrix), testClass)
        NBC_rate += getRecognitionRate(clf_NBC.predict(testMatrix), testClass)
        DTC_rate += getRecognitionRate(clf_DTC.predict(testMatrix), testClass)
        GBDT_rate += getRecognitionRate(clf_GBDT.predict(testMatrix), testClass)
    #输出各个分类器的平均识别率（K个训练集测试集，计算平均）

    print('K Nearest Neighbor mean recognition rate: ', KNN_rate / float(setNums))
    print('Linear Discriminant Analysis mean recognition rate: ', LDA_rate / float(setNums))
    print('Support Vector Machine mean recognition rate: ', SVM_rate / float(setNums))
    print('Logistic Regression mean recognition rate: ', LR_rate / float(setNums))
    print('Random Forest mean recognition rate: ', RF_rate / float(setNums))
    print('Native Bayes Classifier mean recognition rate: ', NBC_rate / float(setNums))
    print('Decision Tree Classifier mean recognition rate: ', DTC_rate / float(setNums))
    print('Gradient Boosting Decision Tree mean recognition rate: ', GBDT_rate / float(setNums))

def totalAlgorithm_2():
    #获取各个分类器
    clf_KNN = KNN()
    clf_LDA = LDA()
    clf_SVM = SVM()
    clf_LR = LR()
    clf_RF = RF()
    clf_NBC = native_bayes_classifier()
    clf_DTC = decision_tree_classifier()
    clf_GBDT = gradient_boosting_classifier()
    #获取训练集和测试集
    # T = getData_2()
    # trainMatrix, trainClass, testMatrix, testClass = T[0], T[1], T[2], T[3]
    #输入训练样本
    clf_KNN.fit(trainMatrix, trainClass)
    clf_LDA.fit(trainMatrix, trainClass)
    clf_SVM.fit(trainMatrix, trainClass)
    clf_LR.fit(trainMatrix, trainClass)
    clf_RF.fit(trainMatrix, trainClass)
    clf_NBC.fit(trainMatrix, trainClass)
    clf_DTC.fit(trainMatrix, trainClass)
    clf_GBDT.fit(trainMatrix, trainClass)
    estimator.get_params()

    #输出各个分类器的识别率
    print('K Nearest Neighbor recognition rate: ', getRecognitionRate(clf_KNN.predict(testMatrix), testClass))
    print('Linear Discriminant Analysis recognition rate: ', getRecognitionRate(clf_LDA.predict(testMatrix), testClass))
    print('Support Vector Machine recognition rate: ', getRecognitionRate(clf_SVM.predict(testMatrix), testClass))
    print('Logistic Regression recognition rate: ', getRecognitionRate(clf_LR.predict(testMatrix), testClass))
    print('Random Forest recognition rate: ', getRecognitionRate(clf_RF.predict(testMatrix), testClass))
    print('Native Bayes Classifier recognition rate: ', getRecognitionRate(clf_NBC.predict(testMatrix), testClass))
    print('Decision Tree Classifier recognition rate: ', getRecognitionRate(clf_DTC.predict(testMatrix), testClass))
    print('Gradient Boosting Decision Tree recognition rate: ', getRecognitionRate(clf_GBDT.predict(testMatrix), testClass))
if __name__ == '__main__':
    # print('K个训练集和测试集的平均识别率')
    # totalAlgorithm_1()
    # print('每类前x%训练，剩余测试，各个模型的识别率')
    # totalAlgorithm_2()
    # selectRFParam()
    # print('随机森林参数调优完成！')
    clf_KNN = KNN()

    print(clf_KNN.get_params())