import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow import keras

if __name__ == '__main__':
    data = pd.read_pickle('esc50_features_short_seg.pkl')
    data['c'] = pd.Categorical(data['c'])
    data['c'] = data.c.cat.codes
    acc = []
    for i in range(1, 6):
        t, f = 0, 0
        model = keras.models.load_model('./esc50_short_' + str(i) + 'fold.h5')
        eval_data = data[data['fold'] == str(i)].drop('fold', axis=1)
        y = eval_data.pop('c').values
        x = np.array([x for x in eval_data.values])
        v_list = []
        for j in tqdm(range(0, len(x))):
            x_eval = np.array([x for x in x[j]])
            result = model(x_eval)
            vote_list = np.zeros(50)
            for r in result:
                r_id = np.where(r == np.max(r))
                vote_list[r_id] += 1
            v = np.where(vote_list == np.max(vote_list))
            v_list.append(v)
            if int(v[0][0]) == int(y[j]):
                t += 1
            else:
                f += 1
        acc.append(t/(t+f))
        print(acc[-1])

    print(acc)