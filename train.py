import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def model(m_name, data):
    if m_name == 'SVM':
        model = svm.SVC(C=0.1, kernel='linear', random_state=930122)
    elif m_name == 'KNN':
        model = KNeighborsClassifier()
    elif m_name == 'RF':
        model = RandomForestClassifier(random_state=930122)

    acc, c_acc = [], []
    for i in range(1, 6):
        train = data[data['fold'] != str(i)].copy()

        model.fit(train.loc[:][feature_col], train['c'])

        valid = data[data['fold'] == str(i)].copy()
        valid['pred'] = model.predict(valid.loc[:][feature_col])

        acc.append(np.sum(valid['c'] == valid['pred'])/float(len(valid['c'])))
        print(m_name, acc[-1])

        category_id = sorted(valid['c'].unique(), key=lambda x: int(x))
        temp = []
        for j in range(0, len(category_id)):
            temp.append(
                len(valid[valid['c'] == str(j)][valid['c'] == valid['pred']]['c'])
                / np.sum(valid['c'] == str(j)))
        c_acc.append(temp)
    category_acc = []
    for i in range(0, len(c_acc[0])):
        _sum = 0
        for j in range(0, len(c_acc)):
            _sum += c_acc[j][i]
        category_acc.append(_sum/len(c_acc))
    return model, acc, category_acc


if __name__ == '__main__':
    esc50 = pd.read_pickle('./features.pkl')
    print(esc50.head(5))
    feature_col = esc50.columns[:-2]

    esc10 = pd.read_csv('./esc50.csv')
    esc10 = esc50.loc[esc10[esc10['esc10']==True]['filename']].copy()
    print(esc10.head(5))

    knn_50, knn_50_acc, knn_50_c_acc = model('KNN', esc50)
    svm_50, svm_50_acc, svm_50_c_acc = model('SVM', esc50)
    rf_50, rf_50_acc, rf_50_c_acc = model('RF', esc50)

    knn_10, knn_10_acc, knn_10_c_acc = model('KNN', esc10)
    svm_10, svm_10_acc, svm_10_c_acc = model('SVM', esc10)
    rf_10, rf_10_acc, rf_10_c_acc = model('RF', esc10)

    esc10_acc = [knn_10_acc, svm_10_acc, rf_10_acc]
    esc50_acc = [knn_50_acc, svm_50_acc, rf_50_acc]

    bw = 0.25
    fig = plt.figure(num=1, figsize=(12, 5))
    plt.subplot(1, 3, 1)
    X = np.arange(5)
    plt.bar(x=X, height=knn_10_acc, width=bw, color='b')
    plt.bar(x=X+bw, height=svm_10_acc, width=bw, color='g', tick_label=np.arange(1, 6))
    plt.bar(x=X+(2*bw), height=rf_10_acc, width=bw, color='r')
    plt.ylim([0.0, 1.0])
    plt.legend(['knn', 'svm', 'rf'])
    plt.xlabel('esc-10')
    plt.ylabel('accuracy')

    plt.subplot(1, 3, 2)
    X = np.arange(5)
    plt.bar(x=X, height=knn_50_acc, width=bw, color='b')
    plt.bar(x=X+bw, height=svm_50_acc, width=bw, color='g', tick_label=np.arange(1, 6))
    plt.bar(x=X+(2*bw), height=rf_50_acc, width=bw, color='r')
    plt.ylim([0.0, 1.0])
    plt.legend(['knn', 'svm', 'rf'])
    plt.xlabel('esc-50')
    plt.ylabel('accuracy')

    plt.subplot(1,3,3)
    X = np.arange(50)
    plt.bar(x=X, height=knn_50_c_acc, width=bw, color='b')
    plt.bar(x=X+bw, height=svm_50_c_acc, width=bw, color='g', tick_label=np.arange(1, 51))
    plt.bar(x=X+(2*bw), height=rf_50_c_acc, width=bw, color='r')
    plt.ylim([0.0, 1.0])
    plt.legend(['knn', 'svm', 'rf'])
    plt.xlabel('esc-50 categorical acc')
    plt.ylabel('accuracy')

    plt.show()
