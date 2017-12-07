# encoding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')
import time

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold


if __name__ == '__main__':

    # load data
    print("loading data...")
    train = pd.read_csv('./data/train_new.csv', header=0)
    test = pd.read_csv('./data/test_new.csv', header=0)
    Y_train = train["Survived"].astype(int)
    X_train = train.drop(labels=["Survived"], axis=1)

    kfold = StratifiedKFold(n_splits=10)
#     '''
#     使用交叉验证法进行模型选择
#     '''
#     random_state = 2 # 很多模型都需要一个随机的设定（比如迭代的初始值等等）。random_state的作用就是固定这个随机设定。调参的时候，这个random_state通常是固定好不变的。
#     classifiers = []
#     classifiers.append(SVC(random_state=random_state))
#     classifiers.append(DecisionTreeClassifier(random_state=random_state))
#     classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state,
#                                           learning_rate=0.1))
#     classifiers.append(RandomForestClassifier(random_state=random_state))
#     classifiers.append(ExtraTreesClassifier(random_state=random_state))
#     classifiers.append(GradientBoostingClassifier(random_state=random_state))
#     classifiers.append(MLPClassifier(random_state=random_state))
#     classifiers.append(KNeighborsClassifier())
#     classifiers.append(LogisticRegression(random_state=random_state))
#     classifiers.append(LinearDiscriminantAnalysis())
#
#     cv_results = []
#     for classifier in classifiers:
#         cv_results.append(cross_val_score(classifier, X_train, y=Y_train, scoring="accuracy", cv=kfold, n_jobs=4))
#     cv_means = []
#     for cv_result in cv_results:
#         cv_means.append(cv_result.mean())
#     # 画图
#     cv_res = pd.DataFrame({"CrossValMeans": cv_means, "Algorithm": ["SVC","DecisionTree","AdaBoost",
# "RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})
#     g = sns.barplot("CrossValMeans", "Algorithm", data=cv_res, palette="Set3", orient="h")
#     g.set_xlabel("Mean Accuracy")
#     g = g.set_title("Cross validation scores")
#     plt.show()

    # 最终选择SVC, AdaBoost, RandomForest , ExtraTrees和GradientBoosting这5个分类器做ensemble

    '''
    调参
    '''
    ### SVC
    print("training SVC...")
    SVMC = SVC(probability=True)
    svc_param_grid = {'kernel': ['rbf'],
                      'gamma': [0.001, 0.01, 0.1, 1],
                      'C': [1, 10, 50, 100, 200, 300, 1000]}
    gsSVMC = GridSearchCV(SVMC, param_grid=svc_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
    gsSVMC.fit(X_train, Y_train)
    SVMC_best = gsSVMC.best_estimator_
    print("SVC best accuracy score：", gsSVMC.best_score_)

    # Adaboost
    print("training Adaboost...")
    DTC = DecisionTreeClassifier()
    adaDTC = AdaBoostClassifier(DTC, random_state=7)
    ada_param_grid = {"base_estimator__criterion": ["gini", "entropy"],
                      "base_estimator__splitter": ["best", "random"],
                      "algorithm": ["SAMME", "SAMME.R"],
                      "n_estimators": [1, 2],
                      "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}
    gsadaDTC = GridSearchCV(adaDTC, param_grid=ada_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
    gsadaDTC.fit(X_train, Y_train)
    ada_best = gsadaDTC.best_estimator_
    print("Adaboost best accuracy score：", gsadaDTC.best_score_)

    # ExtraTrees
    print("training ExtraTrees...")
    ExtC = ExtraTreesClassifier()
    ex_param_grid = {"max_depth": [None],
                     "max_features": [1, 3, 10],
                     "min_samples_split": [2, 3, 10],
                     "min_samples_leaf": [1, 3, 10],
                     "bootstrap": [False],
                     "n_estimators": [100, 300],
                     "criterion": ["gini"]}
    gsExtC = GridSearchCV(ExtC, param_grid=ex_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
    gsExtC.fit(X_train, Y_train)
    ExtC_best = gsExtC.best_estimator_
    print("ExtraTrees best accuracy score：", gsExtC.best_score_)

    # RF
    print("training RF...")
    RFC = RandomForestClassifier()
    rf_param_grid = {"max_depth": [None],
                     "max_features": [1, 3, 10],
                     "min_samples_split": [2, 3, 10],
                     "min_samples_leaf": [1, 3, 10],
                     "bootstrap": [False],
                     "n_estimators": [100, 300],
                     "criterion": ["gini"]}
    gsRFC = GridSearchCV(RFC, param_grid=rf_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
    gsRFC.fit(X_train, Y_train)
    RFC_best = gsRFC.best_estimator_
    print("RF best accuracy score：", gsRFC.best_score_)

    # GBDT
    print("training GBDT...")
    GBC = GradientBoostingClassifier()
    gb_param_grid = {'loss': ["deviance"],
                     'n_estimators': [100, 200, 300],
                     'learning_rate': [0.1, 0.05, 0.01],
                     'max_depth': [4, 8],
                     'min_samples_leaf': [100, 150],
                     'max_features': [0.3, 0.1]
                     }
    gsGBC = GridSearchCV(GBC, param_grid=gb_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
    gsGBC.fit(X_train, Y_train)
    GBC_best = gsGBC.best_estimator_
    print("GBDT best accuracy score：", gsGBC.best_score_)

    '''
    Ensemble modeling
    '''
    print("Ensemble modeling...")
    votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
                                           ('svc', SVMC_best), ('adac', ada_best), ('gbc', GBC_best)], voting='soft',
                               n_jobs=4)
    votingC = votingC.fit(X_train, Y_train)
    # Predict and Submit results
    print("predicting...")
    test_Survived = pd.Series(votingC.predict(test), name="Survived")
    IDtest = pd.read_csv("../input/test.csv")["PassengerId"]
    results = pd.concat([IDtest, test_Survived], axis=1)
    results.to_csv("./data/results.csv", index=False)