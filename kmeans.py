import pandas as pd
from sklearn.cluster import KMeans


# 导入csv文件
file = 'train.csv'
data_train = pd.read_csv(file)
data_train['Source'] = 'train'
file_test = 'test.csv'
data_test = pd.read_csv(file_test)
data_test['Source'] = 'test'
data = pd.concat([data_train, data_test], axis=0)

# 数据预处理
data.loc[data['Sex'] == 'female', 'Sex'] = 0
data.loc[data['Sex'] == 'male', 'Sex'] = 1
data.loc[data['Embarked'].isnull(), 'Embarked'] = 'S'
data_E = pd.get_dummies(data.Embarked)
data = data.drop(['Embarked'], axis=1)
data = data.join(data_E)  # one hot encoder方法处理无序类别
data1 = data.drop(['PassengerId', 'Name', 'Age', 'Cabin', 'Ticket', 'Source'], axis=1)

# kmeans聚类分析
estimator = KMeans(n_clusters=10)  # 构造聚类器
estimator.fit(data1)  # 聚类
class_pred = estimator.labels_   # 获取聚类标签
data['class'] = class_pred
