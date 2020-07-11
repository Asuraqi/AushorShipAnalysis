import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

np.random.seed(1337)
from keras.datasets import  mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,Convolution1D,MaxPooling2D,MaxPooling1D,Flatten
from keras.optimizers import Adam


# # download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# # X shape (60,000 28x28), y shape (10,000, )
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# # data pre-processing，-1 represents the number of samples;1 represents the num of channels,28&28 represents the length,width respectively


def read_data(file):
    data = pd.read_csv(file, header=None)
    X_data, y_data = data.iloc[:, 1:], data.iloc[:, 0]
    return X_data, y_data

TRAIN_FEATURE_FILE = r"data\feature\train__feature_author_50.csv"
TEST_FEATURE_FILE = r"data\feature\test_feature_author_50.csv"

X_train, y_train = read_data(TRAIN_FEATURE_FILE)
X_test, y_test = read_data(TEST_FEATURE_FILE)
# 归一化
X_train = MinMaxScaler().fit_transform(X_train)
X_test = MinMaxScaler().fit_transform(X_test)

# 数值化
labelEncoder = LabelEncoder()
y_train = labelEncoder.fit_transform(y_train)
y_test = labelEncoder.transform(y_test)
X_train = X_train.reshape(-1,1,1,329)  # normalize
X_test = X_test.reshape(-1,1,1,329)    # normalize
y_train = np_utils.to_categorical(y_train,num_classes=50)
y_test = np_utils.to_categorical(y_test, num_classes=50)
#build neural network

model=Sequential(
)

model.add(Convolution1D(
    nb_filter=32,
    # nb_col=5,
    # nb_row=329,
    border_mode='same', #padding method
    kernel_size=32,
    # input_shape=(1,     #channels
    #              1,329) #length and width

))

model.add(Activation('relu'))


model.add(MaxPooling2D(
    pool_size=(1,1),
    strides=(1,1),
    border_mode='same', #padding method
))

# //这是添加第二层神经网络，卷积层，激励函数，池化层
model.add(Convolution1D(64,5,5,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(1,1),border_mode='same'))

# //将经过池化层之后的三维特征，整理成一维。方便后面建立全链接层
model.add(Flatten())
# //1024像素
model.add(Dense(1024))

model.add(Activation('relu'))
# //输出压缩到10维，因为有10个标记
model.add(Dense(50))
# //使用softmax进行分类
model.add(Activation('softmax'))



# Another way to define your optimize

adam=Adam(lr=1e-4)

model.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy'])

print('\nTraining-----------')
model.fit(X_train,y_train,nb_epoch=2,batch_size=32)

print('\nTesting------------')
loss,accuracy=model.evaluate(X_test,y_test)


print('test loss: ', loss)
print('test accuracy: ', accuracy)
