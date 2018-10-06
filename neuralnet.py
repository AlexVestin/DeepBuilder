import keras  # Keras 2.1.2 and TF-GPU 1.8.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import os
import random
from keras import backend as K


# https://stackoverflow.com/questions/46594115/euclidean-distance-loss-function-for-rnn-keras
def euc_dist(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))


def read_line_stripped(f):
    return str(f.readline())[2:]


class NeuralNet:
    def __init__(self):
        self.width = 175
        self.height = 175
        self.nr_colors = 3

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(self.width, self.height, self.nr_colors), activation='relu'))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.2))

        self.model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.2))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(2))
        learning_rate = 0.0001
        opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)

        self.model.compile(loss=euc_dist, optimizer=opt)
        self.tensorboard = TensorBoard(log_dir="logs/stage5",  histogram_freq=0, write_graph=True)

        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []

    def load_training_data(self):
        test_files = os.listdir("training")[:800]
        test_amt = 0.15

        test_size = int(test_amt * len(test_files))

        train_data = []
        for file in test_files:
            img = np.load(os.path.join("training", file))
            position = (int(file.split("_")[0]), int(file.split("_")[1]))
            train_data.append((img, position))

        random.shuffle(train_data)
        train_set = train_data[:-test_size]
        self.train_x = np.array([i[0] for i in train_set]).reshape(-1, self.width, self.height, self.nr_colors)
        self.train_y = np.array([i[1] for i in train_set])

        test_set = train_data[-test_size: -1]
        self.test_x = np.array([i[0] for i in test_set]).reshape(-1, self.width, self.height, self.nr_colors)
        self.test_y = np.array([i[1] for i in test_set])

    def train(self):
        self.model.fit(self.train_x, self.train_y, epochs=10, batch_size=32, validation_data=(self.test_x, self.test_y), shuffle=True, verbose=1, callbacks=[self.tensorboard])


if __name__ == "__main__":
    net = NeuralNet()
    net.load_training_data()
    net.train()