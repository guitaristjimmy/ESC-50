import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow import keras


def major_vote(result, n_class):
    vote_list = np.zeros(n_class)
    for r in result:
        r_id = np.where(r == np.max(r))
        vote_list[r_id] += 1
    return np.where(vote_list == np.max(vote_list))


def probability_vote(result):
    total_prob = np.sum(result, axis=0)
    return np.where(total_prob == np.max(total_prob))


if __name__ == '__main__':
    data = pd.read_pickle('esc50_features_ls.pkl')
    data['c'] = pd.Categorical(data['c'])
    data['c'] = data.c.cat.codes
    acc = []
    for i in range(1, 6):
        tm, fm, tp, fp = 0, 0, 0, 0
        model = keras.models.load_model('./models/esc50_ls/esc50_long_' + str(i) + '_fold.h5')
        eval_data = data[data['fold'] == str(i)].drop('fold', axis=1)
        y = eval_data.pop('c').values
        x = np.array([eval_data.loc[x].dropna() for x in eval_data.index])
        v_list = []
        for j in tqdm(range(0, len(x))):
            x_eval = np.array([x for x in x[j]])
            result = model(x_eval)

            major_v = major_vote(result, 50)
            if int(major_v[0][0]) == int(y[j]):
                tm += 1
            else:
                fm += 1

            prob_v = probability_vote(result)
            if int(prob_v[0][0]) == int(y[j]):
                tp += 1
            else:
                fp += 1

        acc.append([tm/(tm+fm), tp/(tp+fp)])
        print(acc[-1])

    print(acc)

    # plot ------------------------------------------------------------------------------------------------------------
    x = np.arange(0,len(acc))
    plt.bar(x=x, height=acc[:][0], width=0.25, tick_label=[1, 2, 3, 4, 5])
    plt.bar(x=x+0.25, height=acc[:][1], width=0.25)
    plt.ylabel('acc')
    plt.xlabel('fold')
    plt.legend(['MV', 'PV'])
    plt.show()
