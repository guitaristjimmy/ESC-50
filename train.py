import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def model(m_name):
    if m_name == 'SVM':
        model = svm.SVC(C=0.1, kernel='linear', random_state=930122)
    elif m_name == 'KNN':
        model = KNeighborsClassifier()
    elif m_name == 'RF':
        model = RandomForestClassifier(random_state=930122)

    for i in range(1, 6):
        train = data[data['fold'] != str(i)].copy()

        model.fit(train.loc[:][feature_col], train['c'])

        valid = data[data['fold'] == str(i)].copy()
        valid['pred'] = model.predict(valid.loc[:][feature_col])

        print(m_name, np.sum(valid['c'] == valid['pred'])/float(len(valid['c'])))

    return model


if __name__ == '__main__':
    data = pd.read_pickle('./features.pkl')
    feature_col = data.columns[:-2]

    knn = model('KNN')
    svm = model('SVM')
    rf = model('RF')
