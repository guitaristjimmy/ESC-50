import librosa
import os
import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm


class Ext_featrue:
    def __init__(self):
        self.cols = ['mfcc_{}_mean'.format(i) for i in range(1, 14)] +\
                    ['mfcc_{}_std'.format(i) for i in range(1, 14)] +\
                    ['zc_mean', 'zc_std', 'c', 'fold']
        self.features = pd.DataFrame(columns=self.cols)

    def extract_feature(self, path):
        name = path.split('/')[-1]
        audio, sr = librosa.load(path=path, sr=None)
        mel_spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr), ref=np.max)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, S=mel_spectrogram, n_mfcc=13)
        mfcc = np.array([x[1:] for x in mfcc])

        zc = librosa.feature.zero_crossing_rate(y=audio)
        features = [m.mean() for m in mfcc] + [m.std() for m in mfcc] + [zc.mean(), zc.std()]
        df = pd.DataFrame({name: features}).T
        df.columns = self.cols
        df['c'] = name.split('-')[-1][:-4]
        df['fold'] = name.split('-')[0]
        self.features = self.features.append(df)


if __name__ == '__main__':
    root = './ESC-50-master/audio/'
    file_paths = [root+f for f in os.listdir(root)]
    ext = Ext_featrue()
    n_cpu = multiprocessing.cpu_count()
    i = 0
    for fp in tqdm(file_paths):
        ext.extract_feature(fp)
    print(ext.features.head(10))

    ext.features.to_pickle('./features.pkl')