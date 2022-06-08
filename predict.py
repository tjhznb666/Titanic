import pandas as pd
import sklearn.ensemble as ensemble  # ensemble learning: 集成学习
import numpy as np


# 导入csv文件
file = 'data_cleaned.csv'
data = pd.read_csv(file)


# 提取训练数据和预测数据
data1 = data[data['Source'] == 'train']
y = data1['Survived']
X = data1.drop(['Survived', 'Source'], axis=1)
data2 = data[data['Source'] == 'test']
X_p = data2.drop(['Survived', 'Source'], axis=1)


# 模型训练
clf = ensemble.GradientBoostingClassifier()
gbdt_model = clf.fit(X, y)


# 预测
print(X_p.isnull().sum())  # 查看是否有缺失值
y_p = gbdt_model.predict(X_p)
new = []
for x in y_p:
    new.append(int(x))
ll = len(y_p)
lll = np.arange(892, 892+ll)
result_d = {'PassengerId':lll, 'Survived':new}
result = pd.DataFrame(result_d)
result.to_csv('D:\\python_work\\Taitanic\\result.csv', index=False, header=True)
print(result.info())
