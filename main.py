import numpy as np
import pandas as pd
from pandas.plotting import andrews_curves, parallel_coordinates, radviz
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import seaborn as sns
import time

model_dict = {
    'Logistic_Regression': LogisticRegression(),
    'SVM': svm.SVC(),
    "Random_Forest": RandomForestClassifier(),
    "GBDT": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(),
    "Decision_Tree": DecisionTreeClassifier(),
    "Naive_Bayes": GaussianNB()
}

data_frame = pd.read_csv('./mushroom.csv')

# print(data_frame.count())

# feature engineering

for column in data_frame:
    if len(data_frame[column].unique()) == 2:
        data_frame[column] = preprocessing.LabelBinarizer().fit_transform(data_frame[column])
    else:
        data_frame = pd.get_dummies(data_frame, columns=[column])

score_list = []
time_list = []
dimension_range_list = list(range(2, 101))


for key, value in model_dict.items():

    print('==================')
    print('This is {0} model.'.format(key))
    print('==================')

    for new_dimension in dimension_range_list:
        decompositor = PCA(n_components=new_dimension)
        # decompositor = FastICA(n_components=new_dimension, max_iter=200)
        # decompositor = FactorAnalysis(n_components=new_dimension)
        decompositor_result = decompositor.fit_transform(data_frame[data_frame.columns[1::]])
        # print(something_new)
        X = np.array(decompositor_result)
        y = np.array(data_frame[data_frame.columns[0]])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        clf = value
        start_flag = time.time()
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        end_flag = time.time() - start_flag
        print("The score of this classifier is: {0}, and the time is: {1} seconds.".format(score,
                                                                                           end_flag))
        score_list.append(score)
        time_list.append(end_flag)

    score_dataframe = pd.DataFrame(score_list)
    score_dataframe.columns = ['score']
    dimension_dataframe = pd.DataFrame(dimension_range_list)
    dimension_dataframe.columns = ['Dimension']
    time_dataframe = pd.DataFrame(time_list)
    time_dataframe.columns = ['Time']
    dimension_score_dataframe = pd.concat([dimension_dataframe, score_dataframe], axis=1)
    dimension_time_dataframe = pd.concat([dimension_dataframe, time_dataframe], axis=1)

    del (score_list[:])
    del (time_list[:])

    plt.figure()
    sns.pointplot(x='Dimension', y='score', data=dimension_score_dataframe)
    plt.savefig('./results/{0}_dimension_score.png'.format(key))
    plt.show()

    plt.figure()
    sns.pointplot(x='Dimension', y='Time', data=dimension_time_dataframe)
    plt.savefig('./results/{0}_dimension_time.png'.format(key))
    plt.show()
