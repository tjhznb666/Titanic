import pandas as pd


# 导入csv文件
file = 'train.csv'
data_train = pd.read_csv(file)
data_train['Source'] = 'train'
file_test = 'test.csv'
data_test = pd.read_csv(file_test)
data_test['Source'] = 'test'
data = pd.concat([data_train, data_test], axis=0)
data = data.reset_index(drop=True)


# 数据预处理

# 性别数据转化
data.loc[data['Sex'] == 'female', 'Sex'] = 0
data.loc[data['Sex'] == 'male', 'Sex'] = 1
data.loc[data['Cabin'].isnull(), 'Cabin'] = 'No'

# 登岸港口数据转化
data.loc[data['Embarked'].isnull(), 'Embarked'] = 'S'
data_E = pd.get_dummies(data.Embarked)
data = data.drop(['Embarked'], axis=1)
data = data.join(data_E)  # one hot encoder方法处理无序类别

# 年龄数据填充
# 按照头衔对人群分类
data.loc[data['Name'].str.contains('.*Mr[^s].*'), 'Name'] = 'Mr'
data.loc[data['Name'].str.contains('.*Mrs.*'), 'Name'] = 'Mrs'
data.loc[data['Name'].str.contains('.*Miss.*'), 'Name'] = 'Miss'
data.loc[data['Name'].str.contains('.*Master.*'), 'Name'] = 'Master'
data.loc[data['Name'].str.contains('.*Rev.*'), 'Name'] = 'Rev'
data.loc[data['Name'].str.contains('.*Dr.*'), 'Name'] = 'Dr'
data.loc[data['Name'].str.contains('.*Col.*'), 'Name'] = 'Col'
data.loc[~data['Name'].str.contains('.*Mr.*|.*Mrs.*|.*Miss.*|.*Master.*|.*Rev.*|.*Dr.*|.*Col.*'), 'Name'] = 'Else'
# 补充缺失的年龄数据
data_age = data[~data['Age'].isnull()]
data.loc[(data['Age'].isnull()) & (data['Name'] == 'Mr'), 'Age'] = \
    data_age.groupby(['Name'])['Age'].mean()['Mr']  # pandas中,与或非为&|~
data.loc[(data['Age'].isnull()) & (data['Name'] == 'Mrs'), 'Age'] = \
    data_age.groupby(['Name'])['Age'].mean()['Mrs']
data.loc[(data['Age'].isnull()) & (data['Name'] == 'Miss'), 'Age'] = \
    data_age.groupby(['Name'])['Age'].mean()['Miss']
data.loc[(data['Age'].isnull()) & (data['Name'] == 'Master'), 'Age'] = \
    data_age.groupby(['Name'])['Age'].mean()['Master']
data.loc[(data['Age'].isnull()) & (data['Name'] == 'Rev'), 'Age'] = \
    data_age.groupby(['Name'])['Age'].mean()['Rev']
data.loc[(data['Age'].isnull()) & (data['Name'] == 'Dr'), 'Age'] = \
    data_age.groupby(['Name'])['Age'].mean()['Dr']
data.loc[(data['Age'].isnull()) & (data['Name'] == 'Col'), 'Age'] = \
    data_age.groupby(['Name'])['Age'].mean()['Col']
data.loc[(data['Age'].isnull()) & (data['Name'] == 'Else'), 'Age'] = \
    data_age.groupby(['Name'])['Age'].mean()['Else']

# 船舱数据转化
data.loc[data['Cabin'].str.contains('.*A.*'), 'Cabin'] = 'CabinA'
data.loc[data['Cabin'].str.contains('.*B.*'), 'Cabin'] = 'CabinB'
data.loc[data['Cabin'].str.contains('.*C.*'), 'Cabin'] = 'CabinC'
data.loc[data['Cabin'].str.contains('.*D.*'), 'Cabin'] = 'CabinD'
data.loc[data['Cabin'].str.contains('.*E.*'), 'Cabin'] = 'CabinE'
data.loc[data['Cabin'].str.contains('.*F.*'), 'Cabin'] = 'CabinF'
data.loc[data['Cabin'].str.contains('.*G.*'), 'Cabin'] = 'CabinG'
data.loc[data['Cabin'].str.contains('.*No.*'), 'Cabin'] = 'CabinNo'
print(data.groupby(['Cabin']).count()['Pclass'])  # 查看每一种Cabin有多少个数据（选择了没有缺失值的属性Pclass）
data_C = pd.get_dummies(data.Cabin)
data = data.drop(['Cabin'], axis=1)
data = data.join(data_C)  # one hot encoder方法处理无序类别

# 船票价格缺失值填充
data_fare = data[~data['Fare'].isnull()]
data.loc[(data['Fare'].isnull()) & (data['CabinNo'] == 1), 'Fare'] = \
    data_fare.groupby(['CabinNo'])['Fare'].mean()[1]

# 删掉暂时不用的特征
data = data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

# 保存数据
data.to_csv('D:\\python_work\\Taitanic\\data_cleaned.csv', index=False, header=True)
    # index=False,header=True表示不保存行索引,保存列标题
