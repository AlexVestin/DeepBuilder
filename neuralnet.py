import keras  # Keras 2.1.2 and TF-GPU 1.8.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import os
from ast import literal_eval as make_tuple
import random

from keras import backend as K

# https://stackoverflow.com/questions/46594115/euclidean-distance-loss-function-for-rnn-keras
def euc_dist(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))


def read_line_stripped(f):
    return str(f.readline())[2:]


class NeuralNet:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(200, 200, 1), activation='relu'))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2))
        learning_rate = 0.0001
        opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)

        self.model.compile(loss=euc_dist, optimizer=opt)
        self.tensorboard = TensorBoard(log_dir="logs")

        self.train_headers = []
        self.train_x = []
        self.train_y = []

        self.test_x = []
        self.test_y = []
        self.test_headers = []

    def load_training_data(self):
        abs_folder_path = "training"
        test_files = os.listdir(abs_folder_path)[:100]
        test_amt = 0.15

        test_size = int(test_amt * len(test_files))

        train_data = []
        for file in test_files:
            with open(os.path.join( abs_folder_path, file), "r") as f:
                type = read_line_stripped(f)
                position = make_tuple(read_line_stripped(f))
                img = np.fromstring(",".join(line for line in f.readlines()), dtype=np.uint8, sep=",")
                img = img.reshape(200, 200)

                train_data.append((img, (type,), position))

        random.shuffle(train_data)
        train_set = train_data[:-test_size]
        self.train_x = np.array([i[0] for i in train_set]).reshape(-1, 200, 200, 1)
        self.train_headers = np.array([i[1] for i in train_set])
        self.train_y = np.array([i[2] for i in train_set])

        test_set = train_data[-test_size: -1]
        self.test_x = np.array([i[0] for i in test_set]).reshape(-1, 200, 200, 1)
        self.test_headers = np.array([i[1] for i in test_set])
        self.test_y = np.array([i[2] for i in test_set])

        self.predict = np.array(train_data[-1][0]).reshape(-1, 200, 200, 1)
        self.predict_actual = np.array(train_data[-1][2])

    def train(self):
        self.model.fit(self.train_x, self.train_y, batch_size=1, validation_data=(self.test_x, self.test_y), shuffle=True, verbose=1, callbacks=[self.tensorboard])

        prediction = self.model.predict(self.predict, verbose=1)
        print(prediction, self.predict_actual)


if __name__ == "__main__":
    net = NeuralNet()
    net.load_training_data()
    net.train()