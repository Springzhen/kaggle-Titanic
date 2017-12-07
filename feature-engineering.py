# encoding=utf-8

import pandas as pd
import numpy as np
from collections import Counter

if __name__ == '__main__':

    # load data
    train = pd.read_csv('./data/train.csv', header=0)
    test = pd.read_csv('./data/test.csv', header=0)

    # Outlier detection
    def detect_outliers(df, n, features):
        """
        Takes a dataframe df of features and returns a list of the indices
        corresponding to the observations containing more than n outliers according
        to the Tukey method.
        """
        outlier_indices = []

        # iterate over features(columns)
        for col in features:
            # 1st quartile (25%)
            Q1 = np.percentile(df[col], 25)
            # 3rd quartile (75%)
            Q3 = np.percentile(df[col], 75)
            # Interquartile range (IQR)
            IQR = Q3 - Q1

            # outlier step
            outlier_step = 1.5 * IQR
            # Determine a list of indices of outliers for feature col
            outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
            # append the found outlier indices for col to the list of outlier indices
            outlier_indices.extend(outlier_list_col)

        # select observations containing more than 2 outliers
        outlier_indices = Counter(outlier_indices)
        multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
        return multiple_outliers

    Outliers_to_drop = detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])
    # print(train.loc[Outliers_to_drop])
    # Drop outliers
    train = train.drop(Outliers_to_drop, axis=0).reset_index(drop=True)

    # joining train and test set
    train_len = len(train)
    dataset = pd.concat([train, test]).reset_index(drop=True)
    dataset = dataset.fillna(np.nan)

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

    # Fare: Fill Fare missing values with the median value（中位数）
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
    # Apply log to Fare to reduce skewness distribution
    dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

    # Name
    # Get Title from Name
    Name_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
    dataset["Name_Title"] = pd.Series(Name_title)
    # Convert to categorical values Name_Title
    dataset["Name_Title"] = dataset["Name_Title"].replace(
        ['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
        'Rare')
    dataset["Name_Title"] = dataset["Name_Title"].map(
        {"Master": 0, "Ms": 1, "Mme": 1, "Mlle": 1, "Mrs": 1, "Miss": 1, "Mr": 2, "Rare": 3})
    dataset["Name_Title"] = dataset["Name_Title"].astype(int)

    # SibSp and Parch: Create a Family_Size value from SibSp and Parch
    dataset["Family_Size"] = dataset["SibSp"] + dataset["Parch"] + 1
    # Create new feature of family size
    dataset['Single'] = dataset['Family_Size'].map(lambda s: 1 if s == 1 else 0)
    dataset['SmallF'] = dataset['Family_Size'].map(lambda s: 1 if s == 2 else 0)
    dataset['MedF'] = dataset['Family_Size'].map(lambda s: 1 if 3 <= s <= 4 else 0)
    dataset['LargeF'] = dataset['Family_Size'].map(lambda s: 1 if s >= 5 else 0)

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
    dataset["Pclass"] = dataset["Pclass"].astype("category")
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



