# encoding=utf-8

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')

if __name__ == '__main__':

    # load data
    train = pd.read_csv('./data/train.csv', header=0)
    test = pd.read_csv('./data/test.csv', header=0)
    # joining train and test set
    dataset = pd.concat([train, test]).reset_index(drop=True)

    # check for null and missing values
    dataset = dataset.fillna(np.nan)
    # print(dataset.isnull().sum())
    # print(train.info())
    # print(train.isnull().sum())
    # print(train.head())
    # print(train.dtypes)
    # print(train.describe())

    '''
    Numerical values
    '''
    # Correlation matrix between numerical values (SibSp Parch Age Pclass and Fare values) and Survived
    g = sns.heatmap(train[["Survived", "SibSp", "Parch", "Pclass","Age", "Fare"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")

    # Explore SibSp feature vs Survived
    g = sns.factorplot(x="SibSp", y="Survived", data=train, kind="bar", size=6, palette="muted")
    g.despine(left=True) # 去掉图表中的轴
    g = g.set_ylabels("survival probability")

    # Explore Parch feature vs Survived
    g = sns.factorplot(x="Parch", y="Survived", data=train, kind="bar", size=6, palette="muted")
    g.despine(left=True)  # 去掉图表中的轴
    g = g.set_ylabels("survival probability")

    # Explore Pclass feature vs Survived
    g = sns.factorplot(x="Pclass", y="Survived", data=train, kind="bar", size=6, palette="muted")
    g.despine(left=True)  # 去掉图表中的轴
    g = g.set_ylabels("survival probability")

    # Explore Age vs Survived
    g = sns.FacetGrid(train, col='Survived')
    g = g.map(sns.distplot, "Age")
    # Explore Age distibution
    g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade=True)
    g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax=g, color="Blue", shade=True)
    g.set_xlabel("Age")
    g.set_ylabel("Frequency")
    g = g.legend(["Not Survived", "Survived"])

    # Explore Fare vs Survived
    train["Fare"] = train["Fare"].map(lambda i: np.log(i) if i > 0 else 0)  # Apply log to Fare to reduce skewness distribution(偏态分布)
    g = sns.distplot(train["Fare"], color="b", label="Skewness : %.2f" % (train["Fare"].skew()))
    g = g.legend(loc="best")


    '''
    Categorical values
    '''
    # sex
    g = sns.barplot(x="Sex", y="Survived", data=train)
    g = g.set_ylabel("Survival Probability")

    # Embarked
    g = sns.factorplot(x="Embarked", y="Survived", data=train, kind="bar", size=6, palette="muted")
    g.despine(left=True)  # 去掉图表中的轴
    g = g.set_ylabels("survival probability")

    # Explore Pclass vs Embarked
    g = sns.factorplot("Pclass", col="Embarked", data=train, size=6, kind="count", palette="muted")
    g.despine(left=True)
    g = g.set_ylabels("Count")

    plt.show()