# encoding=utf-8

import pandas as pd
import numpy as np

if __name__ == '__main__':

    # load data
    train = pd.read_csv('./data/train.csv', header=0)
    test = pd.read_csv('./data/test.csv', header=0)
    # joining train and test set
    train_len = len(train)
    dataset = pd.concat([train, test]).reset_index(drop=True)
    dataset = dataset.fillna(np.nan)

    # Fare: Fill Fare missing values with the median value（中位数）
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())

    # Embarked: Fill Embarked missing values with the most frequent value（众数）
    dataset["Embarked"] = dataset["Embarked"].fillna(dataset["Embarked"].mode())

    # Sex: convert Sex into categorical value 0 for male and 1 for female
    dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})

    # Age: Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
    index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)
    for i in index_NaN_age:
        age_med = dataset["Age"].median()
        age_pred = dataset["Age"][(
                (dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (
                dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
        if not np.isnan(age_pred):
            dataset['Age'].iloc[i] = age_pred
        else:
            dataset['Age'].iloc[i] = age_med

    # Name
    # Get Title from Name
    Name_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
    dataset["Name_Title"] = pd.Series(Name_title)
    # Convert to categorical values Name_Title
    dataset["Name_Title"] = dataset["Name_Title"].replace(
        ['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
        'Rare')
    dataset["Name_Title"] = dataset["Name_Title"].map(
        {"Master": 0, "Ms": 1, "Mme": 1, "Mlle": 1, "Mrs": 1, "Miss": 2, "Mr": 3, "Rare": 4})
    dataset["Name_Title"] = dataset["Name_Title"].astype(int)

    # SibSp and Parch: Create a Family_Size value from SibSp and Parch
    dataset["Family_Size"] = dataset["SibSp"] + dataset["Parch"] + 1

    # Cabin:  Fill Cabin missing values with 'X',if not ,return the first letter
    dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin']])

    # Ticket: Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X.
    Ticket = []
    for i in list(dataset["Ticket"]):
        if not i.isdigit():
            Ticket.append(i.replace(".", "").replace("/", "").strip().split(' ')[0])  # Take prefix
        else:
            Ticket.append("X")
    dataset["Ticket"] = Ticket

    # Convert Name_Title, Embarked , Cabin, Ticket and Pclass into dummy variables(进行哑编码)
    dataset = pd.get_dummies(dataset, columns=["Name_Title"], prefix="Nt")
    dataset = pd.get_dummies(dataset, columns=["Embarked"], prefix="Em")
    dataset = pd.get_dummies(dataset, columns=["Cabin"], prefix="Ca")
    dataset = pd.get_dummies(dataset, columns=["Ticket"], prefix="Tc")
    dataset = pd.get_dummies(dataset, columns=["Pclass"], prefix="Pc")

    # drop PassengerId and Name
    dataset.drop(labels=["PassengerId"], axis=1, inplace=True) # inplace=True表示直接在原dataset中进行删除
    dataset.drop(labels=["Name"], axis=1, inplace=True)


    # Writing to a csv file
    train = dataset[:train_len]
    test = dataset[train_len:]
    test.drop(labels=["Survived"], axis=1, inplace=True)
    train.to_csv('./data/train_new.csv')
    test.to_csv('./data/test_new.csv')



