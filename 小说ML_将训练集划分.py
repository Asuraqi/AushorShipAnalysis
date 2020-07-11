"""
    导包
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import GridSearchCV
"""
    读取数据
"""
def read_data(file):
    data = pd.read_csv(file, header=None)
    X_data, y_data = data.iloc[:, 1:], data.iloc[:, 0]
    return X_data, y_data

TRAIN_FEATURE_FILE = r"data\feature\train_feature_20author.csv"
X_data, y_data = read_data(TRAIN_FEATURE_FILE)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

# 归一化
X_train = MinMaxScaler().fit_transform(X_train)
X_test = MinMaxScaler().fit_transform(X_test)

# 数值化
labelEncoder = LabelEncoder()
y_train = labelEncoder.fit_transform(y_train)
y_test = labelEncoder.transform(y_test)


"""
    SVM训练和评估
"""
clf = SVC(C=0.8, kernel='poly', gamma=20, decision_function_shape='ovr')
clf.fit(X_train, y_train)

# 评估
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

"""
    保存模型
"""
joblib.dump(clf, "train_split_model_svm")

