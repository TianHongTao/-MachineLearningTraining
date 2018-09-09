import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


if __name__ == '__main__':

    data_train = pd.read_csv("D://Training/titanic/train.csv")
    data_train.info()
    data_train.describe()
    # fig = plt.figure()
    # fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

##初步画图
    # plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
    # data_train.Survived.value_counts().plot(kind='bar')# 柱状图
    # plt.title(u"获救情况 (1为获救)") # 标题
    # plt.ylabel(u"人数")
    #
    # plt.subplot2grid((2,3),(0,1))
    # data_train.Pclass.value_counts().plot(kind="bar")
    # plt.ylabel("人数")
    # plt.title("乘客等级分布")
    #
    # plt.subplot2grid((2,3),(0,2))
    # data_train.Age[data_train.Survived == 1].plot(kind='kde')
    # data_train.Age[data_train.Survived == 0].plot(kind='kde')
    # plt.ylabel("人数")# 设定纵坐标名称
    # plt.xlabel("年龄")
    # plt.legend(('存活','丧生'),loc='best')
    # plt.title("按年龄看获救分布 (1为获救)")
    #
    #
    # plt.subplot2grid((2,3),(1,0), colspan=2)
    # data_train.Age.plot(kind='kde')
    # plt.xlabel("年龄")# plots an axis lable
    # plt.ylabel("密度")
    # plt.title("各等级的乘客年龄分布")
    #
    #
    # plt.subplot2grid((2,3),(1,2))
    # data_train.Embarked.value_counts().plot(kind='bar')
    # plt.title("各登船口岸上船人数")
    # plt.ylabel("人数")


#是否获救与乘客等级的关系
    # Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
    # Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
    # df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
    # df.plot(kind='bar', stacked=True)
    # plt.title("各乘客等级的获救情况")
    # plt.xlabel("乘客等级")
    # plt.ylabel("人数")

#是否获救与年龄的关系
    # Survived_0 = data_train.Age[data_train.Survived == 0].value_counts()
    # Survived_1 = data_train.Age[data_train.Survived == 1].value_counts()
    # df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
    # df.plot(kind='bar', stacked=True)
    # plt.title("各乘客年龄的获救情况")
    # plt.xlabel("乘客年龄")
    # plt.ylabel("人数")

#是否获救与上船位置
    # Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
    # Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
    # df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
    # df.plot(kind='bar', stacked=True)
    # plt.title("各乘客上船位置的获救情况")
    # plt.xlabel("乘客上传位置")
    # plt.ylabel("人数")
    #
    # plt.show()

#使用随机森林补全年龄
    age_df = data_train[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
    age_df_notnull = age_df.loc[(data_train['Age'].notnull())]
    age_df_isnull = age_df.loc[(data_train['Age'].isnull())]
    X = age_df_notnull.values[:,1:]
    Y = age_df_notnull.values[:,0]

    RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
    RFR.fit(X,Y)
    predictAges = RFR.predict(age_df_isnull.values[:,1:])
    data_train.loc[data_train['Age'].isnull(), ['Age']]= predictAges
    data_train.loc[(data_train.Cabin.notnull()), 'Cabin'] = "Yes"
    data_train.loc[(data_train.Cabin.isnull()), 'Cabin'] = "No"

    dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

    df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    scaler = preprocessing.StandardScaler()
    data_train['Age_scaled'] = scaler.fit_transform(data_train['Age'].values.reshape(-1, 1))
    data_train['Fare_scaled'] = scaler.fit_transform(data_train['Fare'].values.reshape(-1, 1))

#对测试数据进行补全调整
    data_test = pd.read_csv("D://Training/titanic/test.csv")
    data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0

    age_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    age_df_notnull = age_df.loc[(data_test['Age'].notnull())]
    age_df_isnull = age_df.loc[(data_test['Age'].isnull())]
    X = age_df_notnull.values[:,1:]
    Y = age_df_notnull.values[:,0]

    RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
    RFR.fit(X,Y)
    predictAges = RFR.predict(age_df_isnull.values[:,1:])
    data_test.loc[data_test['Age'].isnull(), ['Age']]= predictAges
    data_test = set_Cabin_type(data_test)
    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

    df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    data_test['Age_scaled'] = scaler.fit_transform(data_test['Age'].values.reshape(-1, 1))
    data_test['Fare_scaled'] = scaler.fit_transform(data_test['Fare'].values.reshape(-1, 1))


