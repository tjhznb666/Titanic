import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.ensemble as ensemble  # ensemble learning: 集成学习
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


# 导入csv文件
file = 'data_cleaned.csv'
data = pd.read_csv(file)


# 选择训练集和测试集
a = (data['Source'] == 'train')
print(a)
data1 = data[data['Source'] == 'train']
y = data1['Survived']
X = data1.drop(['Survived', 'Source'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

clf = ensemble.GradientBoostingClassifier()
gbdt_model = clf.fit(X_train, y_train)

# 使用GBDT对测试集进行预测
# 得到测试集每个样本为正例的概率（在这里用的是predict_proba，其他模型中也可以用decision_function）
y_score = gbdt_model.predict_proba(X_test)[:, 1]
y_pre = gbdt_model.predict(X_test)
print('GBDT精确度...')
print(metrics.classification_report(y_test, y_pre))
print('GBDT AUC...')
fpr, tpr, th = metrics.roc_curve(y_test, y_score)  # 构造 roc 曲线，第三个输出为阈值（每个阈值对应一个DPR和TPR）
ks = max(tpr - fpr)
print('KS=',ks)
print('AUC = %.4f' %metrics.auc(fpr, tpr))  # 求AUC值(ROC曲线下方面积)

# 画ROC曲线
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()
