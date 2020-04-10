import librosa
import os
import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm


class Ext_feature:
    def __init__(self):
        self.features = pd.DataFrame()

    def extract_esc50_feature(self, path, size):
        name = path.split('/')[-1]
        audio, sr = librosa.load(path=path, sr=22050)
        audio = librosa.util.normalize(audio)
        aug = [audio]
        for _ in range(0, 3):
            aug.append(np.roll(audio, int(sr/10)*np.random.randint(0, 20)))
        for i, audio in enumerate(aug):
            mel_spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, n_mels=60))
            segments = self.divide2segment(mel_spectrogram, size=size)

            df = pd.DataFrame({name+'_'+str(i): segments}).T
            df.columns = ['segment' + '-' + str(i) for i in range(0, len(segments))]
            df['c'] = name.split('-')[-1][:-4]
            df['fold'] = name.split('-')[0]
            self.features = self.features.append(df)

    def extract_esc10_feature(self, path, size):
        name = path.split('/')[-1]
        audio, sr = librosa.load(path=path, sr=22050)
        aug = [audio]
        for _ in range(0, 3):
            aug.append(np.roll(audio, int(sr/10)*np.random.randint(0, 20)))

        ts, ps = False, False
        target = name.split('-')[-1][:-4]
        if target in ['0', '1', '4', '5', '41']:
            ts, ps = True, True
        elif target in ['11', '40']:
            ts = True

        if ts and ps:
            aug.append(librosa.effects.time_stretch(librosa.effects.pitch_shift(audio, sr, np.random.randint(-2, 3)),
                                                    np.random.random()*0.15+0.95))
        elif ts or ps:
            aug.append(librosa.effects.time_stretch(audio, np.random.random()*0.15+0.95))

        for i, audio in enumerate(aug):
            mel_spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, n_mels=60))
            segments = self.divide2segment(mel_spectrogram, size=size)

            df = pd.DataFrame({name+'_'+str(i): segments}).T
            df.columns = ['segment' + '-' + str(i) for i in range(0, len(segments))]
            df['c'] = name.split('-')[-1][:-4]
            df['fold'] = name.split('-')[0]
            self.features = self.features.append(df)

    def divide2segment(self, spectrogram, size):
        segments = []
        if size == 41:
            for s in range(0, len(spectrogram[0]), int(size/2)):
                seg = spectrogram[:, s:s+size]
                if len(seg[0]) < size:
                    seg = np.hstack((seg, np.zeros((seg.shape[0], size-len(seg[0])))))
                if np.mean(seg) > -50:
                    delta = librosa.feature.delta(seg)
                    s = np.dstack((seg, delta))
                    segments.append(s)
        else:
            for s in range(0, len(spectrogram[0]), int(size*0.1)):
                seg = spectrogram[:, s:s+size]
                if len(seg[0]) < size:
                    seg = np.hstack((seg, np.zeros((seg.shape[0], size-len(seg[0])))))
                if np.mean(seg) > -50:
                    delta = librosa.feature.delta(seg)
                    s = np.dstack((seg, delta))
                    segments.append(s)
        return segments


if __name__ == '__main__':
    root = './ESC-50-master/audio/'
    ext = Ext_feature()

    """
    ESC - 50
    """
    # file_paths = [root+f for f in os.listdir(root)]
    # for fp in tqdm(file_paths):
    #     ext.extract_esc50_feature(fp, 101)  # size = 41 :: short segment / size = 101 :: long segment
    # print(ext.features)
    # ext.features.to_pickle('./esc50_features_ls.pkl')

    """
    ESC - 10
    """
    meta = pd.read_csv('esc50.csv', index_col=0)
    print(meta.head(5))
    file_paths = [root+f for f in meta[meta['esc10']==True].index]
    print(file_paths[:5])

    for fp in tqdm(file_paths):
        ext.extract_esc10_feature(fp, 41)  # size = 41 :: short segment / size = 101 :: long segment
    print(ext.features.head(5))
    ext.features.to_pickle('./esc10_features_ss.pkl')
