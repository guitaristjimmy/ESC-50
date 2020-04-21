import librosa
import os
import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm


class Ext_feature:
    def __init__(self):
        self.esc50_features = pd.DataFrame()
        self.esc10_features = pd.DataFrame()
        self.urban8k_features = []
        self.urban8k_class = []
        self.urban8k_fold = []

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
            self.esc50_features = self.esc_50features.append(df)

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
            df.columns = ['segment' + '-' + str(j) for j in range(0, len(segments))]
            df['c'] = name.split('-')[-1][:-4]
            df['fold'] = name.split('-')[0]
            self.esc10_features = self.esc10_features.append(df)

    def extract_urban8k_feature(self, path, size, fold):
        name = path.split('/')[-1]
        audio, sr = librosa.load(path=path, sr=22050)
        aug = [audio]
        for _ in range(0, 3):
            aug.append(np.roll(audio, int(sr/10)*np.random.randint(0, 20)))

        ts_ps = False, False
        target = name.split('-')[1]
        if target in ['1', '2', '3', '6', '8', '9']:
            ts_ps = True, True

        if ts_ps:
            aug.append(librosa.effects.time_stretch(librosa.effects.pitch_shift(audio, sr, np.random.randint(0, 2)),
                                                    np.random.random()*0.15+0.95))

        for i, audio in enumerate(aug):
            mel_spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, n_mels=60))
            segments = self.divide2segment(mel_spectrogram, size=size)

            self.urban8k_features.append(segments)
            self.urban8k_class.append(name.split('-')[1])
            self.urban8k_fold.append(fold)

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


def urban8k_ext_multi(paths, fold):
    ext_f = Ext_feature()
    for fp in tqdm(paths, desc=str(fold)+'processing'):
        ext_f.extract_urban8k_feature(path=fp, size=101, fold=str(fold))  # size = 41 :: short segment / size = 101 :: long segment
    df = pd.DataFrame(data=ext_f.urban8k_features)
    df['c'] = ext_f.urban8k_class
    df['fold'] = ext_f.urban8k_fold
    print(df.head(5))
    df.to_pickle('./dataset/urban8k_ls/urban8k_features_'+str(fold)+'_ls.pkl')


if __name__ == '__main__':
    root = './ESC-50-master/audio/'
    ext = Ext_feature()

    """
    ESC - 10
    """
    # meta = pd.read_csv('esc50.csv', index_col=0)
    # print(meta.head(5))
    # file_paths = [root+f for f in meta[meta['esc10']==True].index]
    # print(file_paths[:5])
    #
    # for fp in tqdm(file_paths):
    #     ext.extract_esc10_feature(fp, 101)  # size = 41 :: short segment / size = 101 :: long segment
    # print(ext.esc10_features.head(5))
    # ext.esc10_features.to_pickle('./esc10_features_ls.pkl')

    """
    ESC - 50
    """
    # file_paths = [root+f for f in os.listdir(root)]
    # for fp in tqdm(file_paths):
    #     ext.extract_esc50_feature(fp, 101)  # size = 41 :: short segment / size = 101 :: long segment
    # print(ext.esc50_features)
    # ext.esc50_features.to_pickle('./esc50_features_ls.pkl')

    """
    Urban Sound 8k
    """
    root = './urbansound8k/audio/'
    file_paths = []
    for fold in os.listdir(root):
        if not fold.endswith('.DS_Store'):
            file_paths.append([])
            for file in os.listdir(root+fold):
                if file.endswith('.wav'):
                    file_paths[-1].append(root+fold+'/'+file)
    i = 0
    while i < len(file_paths):
        process = []
        num_cpu = multiprocessing.cpu_count()
        for j in range(0, int(num_cpu/2)):
            process.append(multiprocessing.Process(target=urban8k_ext_multi, args=(file_paths[i], i+1)))
            print('\nprocess'+str(j+1)+'start')
            process[-1].start()
            i += 1
            if i >= len(file_paths):
                break
        for j, p in enumerate(process):
            p.join()
            print('\nprocess'+str(j+1)+'join')
            p.close()
    for i in range(0, len(file_paths)):
        for fp in tqdm(file_paths[i]):
            ext.extract_urban8k_feature(path=fp, size=41, fold=str(i+1))  # size = 41 :: short segment / size = 101 :: long segment
        df = pd.DataFrame(data=ext.urban8k_features)
        df['c'] = ext.urban8k_class
        df['fold'] = ext.urban8k_fold
        print(df.head(5))
        df.to_pickle('./dataset/urban8k_ss/urban8k_features_'+str(i)+'_ss.pkl')
        ext.urban8k_class = []
        ext.urban8k_features = []
        ext.urban8k_fold = []
