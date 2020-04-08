import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers


class esc50_convnet:
    def __init__(self):
        pass

    def data_load(self, path):
        data = pd.read_pickle(path)
        data['c'] = pd.Categorical(data['c'])
        data['c'] = data.c.cat.codes

        return data

    def model(self, learning_rate, n_class, in_shape=(60, 41, 2)):
        model = keras.Sequential()
        model.add(layers.Conv2D(filters=80, kernel_size=(57, 6), strides=(1, 1),
                                activation='relu', kernel_regularizer=keras.regularizers.l2(0.001),
                                input_shape=in_shape))
        model.add(layers.Dropout(0.5))
        model.add(layers.MaxPooling2D(pool_size=(4, 3), strides=(1, 3)))
        model.add(layers.Conv2D(filters=80, kernel_size=(1, 3), strides=(1, 1), activation='relu',
                                kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
        model.add(layers.Flatten())
        model.add(layers.Dense(5000, activation='relu',
                               kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(5000, activation='relu',
                               kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_class, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.001)))

        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.002, momentum=0.9, nesterov=True),
                      loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        return model


class MyCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MyCallBack, self).__init__()
        self.val_accuracy_logs, self.train_accuracy_logs = [], []
        self.epoch_num = 0

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            self.val_accuracy_logs.append([])
            self.train_accuracy_logs.append([])
        self.epoch_num += 1

    def on_epoch_end(self, epoch, logs=None):
        self.val_accuracy_logs[-1].append(logs['val_categorical_accuracy'])
        self.train_accuracy_logs[-1].append(logs['categorical_accuracy']*100)

    # def on_train_batch_end(self, batch, logs=None):
    #     print('\t epoch :: ', self.epoch_num)


if __name__ == '__main__':
    esc50 = esc50_convnet()
    my_callback = MyCallBack()
    data = esc50.data_load('./esc50_features_short_seg.pkl')
    bs = 256

    for i in range(2, 6):
        model = esc50.model(learning_rate=0.002, n_class=50, in_shape=(60, 41, 2))

        train_df = data[data['fold'] != str(i)].drop('fold', axis=1).fillna(False)
        val_df = data[data['fold'] == str(i)].drop('fold', axis=1).fillna(False)

        y = train_df.pop('c')
        x_train, y_train = [], []
        for xid in tqdm(train_df.index):
            x = train_df.loc[xid]
            for seg in x:
                if seg is not False:
                    x_train.append(seg)
                    y_train.append(y.loc[xid])
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        y_train = keras.utils.to_categorical(np.array(y_train))

        y = val_df.pop('c')
        x_valid, y_valid = [], []
        for xid in tqdm(val_df.index):
            x = val_df.loc[xid]
            for seg in x:
                if seg is not False:
                    x_valid.append(seg)
                    y_valid.append(y.loc[xid])
        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)
        y_valid = keras.utils.to_categorical(np.array(y_valid))

        model.fit(x=x_train, y=y_train, validation_data=(x_valid, y_valid), batch_size=bs, epochs=300,
                  verbose=2, shuffle=True, callbacks=[my_callback])
        model.save('./esc50_short_'+str(i)+'fold.h5')

    # Plot -------------------------------------------------------------------------------------------------------------
    fig = plt.figure()
    for i in range(0, 5):
        plt.subplot(1, 5, i+1)
        plt.plot(my_callback.val_accuracy_logs[i], linestyle=':')
        plt.plot(my_callback.train_accuracy_logs[i])
        plt.legend(['val_acc', 'train_acc'])
        plt.xlabel('epoch')
    plt.show()
