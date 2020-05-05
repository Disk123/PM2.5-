'''
    1、train.csv文中，包含了台湾包含台湾丰原地区240天的气象观测资料(取每个月前20天
    的数据做训练集，12月X20天=240天，每月后10天数据用于测试，对学生不可见)
    2、每天的监测时间点为0时，1时......到23时，共24个时间节点。
    3、每天的检测指标包括CO、NO、PM2.5、PM10等气体浓度，是否降雨、刮风等气象信息，共计18项

    为了便于测试预测模型。
    使用连续8天的气象观测数据，来预测第9天的PM2.5含量。根据李宏毅老师采用热成像法和散点图法对观测数据分析，可以得知
    PM2.5、PM10、SO2与PM2.5的预测存在着较大联系，因此将使用这三种属性来预测第9天的PM2.5值。
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 指定相对路径
path = "./Dataset/"

# 利用 pands 进行读取文件操作
train = pd.read_csv('train.csv', engine='python', encoding='utf-8')
test = pd.read_csv('test.csv', engine='python', encoding='gbk')

# 抽取PM2.5  PM10   SO2作为预测PM2.5的输入属性数据
#  训练数据
train_pm25 = train[train['測項'] == 'PM2.5']
train_pm10 = train[train['測項']=='PM10']
train_so2 = train[train['測項']=='SO2']
#  测试数据
test_pm25 = test[test['AMB_TEMP'] == 'PM2.5']
test_pm10 = test[test['AMB_TEMP'] == 'PM10']
test_so2 = test[test['AMB_TEMP'] == 'SO2']

# 删除无关特征
train_pm25 = train_pm25.drop(['日期', '測站','測項'], axis=1)
train_pm10 = train_pm10.drop(['日期', '測站','測項'], axis=1)
train_so2 = train_so2.drop(['日期', '測站','測項'], axis=1)


#  测试数据的标签
test_y=test_pm25.iloc[:,10]
#  测试数据的输入数据
test_pm25 = test_pm25.iloc[:, 2:10]
test_pm10 = test_pm10.iloc[:, 2:10]
test_so2 = test_so2.iloc[:, 2:10]


train_pm25_x = []   #暂时存储训练数据的PM2.5数据
train_pm25_y = []   #暂时存储训练数据的pm2.5标签

train_pm10_x = []   # 暂时存储训练数据的PM10数据
train_so2_x = []    #  暂时存储训练数据so2数据


for i in range(16):
    # 抽取pm2.5的数据
    x_pm25 = train_pm25.iloc[:, i:i + 8]              #  取出每相隔9列的数据
    # notice if we don't set columns name, it will have different columns name in each iteration
    x_pm25.columns = np.array(range(8))          #   每一列的Index设置成为0～8 ，一共9个index
    y_pm25 = train_pm25.iloc[:, i + 8]                #从第十列开始取，设置为x对应的标签
    y_pm25.columns = np.array(range(1))          #设置index
    train_pm25_x.append(x_pm25)
    train_pm25_y.append(y_pm25)
    #抽取pm10数据
    x_pm10 = train_pm10.iloc[:, i:i + 8]  # 取出每相隔9列的数据
    x_pm10.columns = np.array(range(8))  # 每一列的Index设置成为0～8 ，一共9个index
    train_pm10_x.append(x_pm10)
    #抽取so2数据
    x_so2 = train_so2.iloc[:, i:i + 8]  # 取出每相隔9列的数据
    x_so2.columns = np.array(range(8))  # 每一列的Index设置成为0～8 ，一共9个index
    train_so2_x.append(x_so2)


# review "Python for Data Analysis" concat操作
# train_x and train_y are the type of Dataframe
# 取出 PM2.5 的数据，训练集中一共有 240 天，每天取出 16 组 含有 8 个特征 和 1 个标签的数据，共有 240*16*8个数据

#此时的x的内部结构应该是[d0,d1,d2....d15],而对应每个d0内部则是240*9
#  经过pd.concat之后，d0、d1、d2...融合到一起，因此此时的x的内部的shape变为{（240*16），8}  y同理可得
train_pm25_x = pd.concat(train_pm25_x) # (3840, 8) Dataframe类型
train_pm25_y = pd.concat(train_pm25_y)

train_pm10_x = pd.concat(train_pm10_x) # (3840, 8) Dataframe类型
train_so2_x = pd.concat(train_so2_x) # (3840, 8) Dataframe类型


# 将str数据类型转换为 numpy的 ndarray 类型
# 训练数据的标签
train_pm25_y = np.array(train_pm25_y, float)
#  测试数据的标签
test_pm25_y = np.array(test_y,float)
#  测试数据
test_pm25_x = np.array(test_pm25, float)
test_pm10_x = np.array(test_pm10, float)
test_so2_x = np.array(test_so2, float)


# 进行标准缩放，即数据归一化，加快算法的收敛速度
#  StandardScaler是进行归一化的函数
ss = StandardScaler()
# 训练数据拟合归一化
ss.fit(train_pm25_x)
train_pm25_x = ss.transform(train_pm25_x)
ss.fit(train_pm10_x)
train_pm10_x = ss.transform(train_pm10_x)
ss.fit(train_so2_x)
train_so2_x = ss.transform(train_so2_x)
#  测试数据归一化
ss.fit(test_pm25_x)
test_pm25_x = ss.transform(test_pm25_x)
ss.fit(test_pm10_x)
test_pm10_x = ss.transform(test_pm10_x)
ss.fit(test_so2_x)
test_so2_x = ss.transform(test_so2_x)

#  np.hstack合并pm2.5 pm10 so2三中属性的 因此训练数据的维度变为24
train_x_24 = np.hstack((train_pm25_x,train_pm10_x,train_so2_x))
test_x_24 = np.hstack((test_pm25_x,test_pm10_x,test_so2_x))
train_y_24 = train_pm25_y
test_y_24 = test_pm25_y

#  保存训练数据和测试数据
np.save("train_x_24.npy",train_x_24)
np.save("train_y_24.npy",train_y_24)
np.save("test_x_24.npy",test_x_24)
np.save("test_y_24.npy",test_y_24)

print('断点')