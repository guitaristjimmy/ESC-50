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
        audio, sr = librosa.load(path=path, sr=None)
        audio = librosa.util.normalize(audio)
        aug = [audio]
        for _ in range(0, 3):
            aug.append(np.roll(audio, int(sr/10)*np.random.randint(0, 20)))
        for i, audio in enumerate(aug):
            mel_spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, n_mels=60),
                                                  ref=np.max)
            delta = librosa.feature.delta(mel_spectrogram)
            f = np.dstack((mel_spectrogram, delta))
            features = self.divide2segment(f, size=size)
            df = pd.DataFrame({name+'_'+str(i): features}).T
            df.columns = ['segment' + '-' + str(i) for i in range(0, len(features))]
            # df = df.T
            # df = df.set_index('name')
            df['c'] = name.split('-')[-1][:-4]
            df['fold'] = name.split('-')[0]
            self.features = self.features.append(df)

    def extract_esc10_feature(self, path, size):
        name = path.split('/')[-1]
        audio, sr = librosa.load(path=path, sr=None)
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

        for audio in aug:
            mel_spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, n_mels=60),
                                                  ref=np.max)
            delta = librosa.feature.delta(mel_spectrogram)
            f = np.dstack((mel_spectrogram, delta))
            features = self.divide2segment(f, size=size)
            df = pd.DataFrame({name: features}).T
            df['c'] = target
            df['fold'] = name.split('-')[0]
            self.features = self.features.append(df)

    def divide2segment(self, spectrogram, size):
        segments = []
        if size == 41:
            for s in range(0, len(spectrogram[0]), int(size/2)):
                seg = spectrogram[:, s:s+size, :]
                if len(seg[0]) < size:
                    seg = np.hstack((seg, np.zeros((seg.shape[0], size-len(seg[0]), seg.shape[2]))))
                if np.mean(seg) > -70:
                    segments.append(seg)
        else:
            for s in range(0, len(spectrogram[0])/size*(10/9), int(size/2)):
                segments.append(spectrogram[:][s:s+size])
        return segments


if __name__ == '__main__':
    root = './ESC-50-master/audio/'
    file_paths = [root+f for f in os.listdir(root)]
    ext = Ext_feature()
    for fp in tqdm(file_paths):
        ext.extract_esc50_feature(fp, 41)
    print(ext.features)
    ext.features.to_pickle('./esc50_features_short_seg.pkl')
