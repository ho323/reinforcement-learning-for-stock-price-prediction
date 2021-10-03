'''
속성
shared_network: 신경망의 상단부로 여러 신경망이 공유할 수 있음(예를 들어, A2C에서는 가치 신경망과 정책 신경망이 신경망의 상단부를 공유하고 하단 부분만 가치 예측과 확률 예측을 위해 달라짐)
activation: 신경망의 출력 레이어 활성화 함수 이름("linear", "sigmoid" 등)
loss: 신경망의 손실 함수
lr: 신경망의 학습 속도
model: Keras 라이브러리로 구성한 최종 신경망 모델

함수
predict(): 신경망을 통해 투자 행동별 가치나 확률 계산
train_on_batch(): 배치 학습을 위한 데이터 생성
save_model(): 학습한 신경망을 파일로 저장
load_model(): 파일로 저장한 신경망을 로드
get_shared_network(): 신경망의 상단부를 생성하는 클래스 함수
'''

import os
import threading
import numpy as np


if os.environ['KERAS_BACKEND'] == 'tensorflow':
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, \
        BatchNormalization, Dropout, MaxPooling2D, Flatten
    from tensorflow.keras.optimizers import SGD
    import tensorflow as tf
    tf.compat.v1.disable_v2_behavior()
    print('Eager Mode: {}'.format(tf.executing_eagerly()))
elif os.environ['KERAS_BACKEND'] == 'plaidml.keras.backend':
    from keras.models import Model
    from keras.layers import Input, Dense, LSTM, Conv2D, \
        BatchNormalization, Dropout, MaxPooling2D, Flatten
    from keras.optimizers import SGD


class Network:
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, lr=0.001, 
                shared_network=None, activation='sigmoid', loss='mse'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.shared_network = shared_network
        self.activation = activation
        self.loss = loss
        self.model = None

    def predict(self, sample):
        with self.lock:
            return self.model.predict(sample).flatten()

    def train_on_batch(self, x, y):
        loss = 0.
        with self.lock:
            history = self.model.fit(x, y, epochs=10, verbose=False)
            loss += np.sum(history.history['loss'])
        return loss

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)

    @classmethod
    def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0):
        if net == 'dnn':
            return DNN.get_network_head(Input((input_dim,)))
        elif net == 'lstm':
            return LSTMNetwork.get_network_head(
                Input((num_steps, input_dim)))
        elif net == 'cnn':
            return CNN.get_network_head(
                Input((1, num_steps, input_dim)))


class DNN(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inp = None
        output = None
        if self.shared_network is None:
            inp = Input((self.input_dim,))
            output = self.get_network_head(inp).output
        else:
            inp = self.shared_network.input
            output = self.shared_network.output
        output = Dense(
            self.output_dim, activation=self.activation, 
            kernel_initializer='random_normal')(output)
        self.model = Model(inp, output)
        self.model.compile(
            optimizer=SGD(learning_rate=self.lr), loss=self.loss)

    @staticmethod
    def get_network_head(inp):
        output = Dense(256, activation='sigmoid', 
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(128, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(64, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(32, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        return Model(inp, output)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.input_dim))
        return super().predict(sample)
    

class LSTMNetwork(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps
        inp = None
        output = None
        if self.shared_network is None:
            inp = Input((self.num_steps, self.input_dim))
            output = self.get_network_head(inp).output
        else:
            inp = self.shared_network.input
            output = self.shared_network.output
        output = Dense(
            self.output_dim, activation=self.activation, 
            kernel_initializer='random_normal')(output)
        self.model = Model(inp, output)
        self.model.compile(
            optimizer=SGD(learning_rate=self.lr), loss=self.loss)

    @staticmethod
    def get_network_head(inp):
        # cuDNN 사용을 위한 조건
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
        output = LSTM(256, dropout=0.1, return_sequences=True, stateful=False, kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = LSTM(128, dropout=0.1, return_sequences=True, stateful=False, kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(64, dropout=0.1, return_sequences=True, stateful=False, kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(32, dropout=0.1, stateful=False, kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        return Model(inp, output)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.num_steps, self.input_dim))
        return super().predict(sample)


class CNN(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps
        inp = None
        output = None
        if self.shared_network is None:
            inp = Input((self.num_steps, self.input_dim, 1))
            output = self.get_network_head(inp).output
        else:
            inp = self.shared_network.input
            output = self.shared_network.output
        output = Dense(
            self.output_dim, activation=self.activation,
            kernel_initializer='random_normal')(output)
        self.model = Model(inp, output)
        self.model.compile(
            optimizer=SGD(learning_rate=self.lr), loss=self.loss)

    @staticmethod
    def get_network_head(inp):
        output = Conv2D(256, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(64, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(32, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Flatten()(output)
        return Model(inp, output)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim, 1))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape(
            (-1, self.num_steps, self.input_dim, 1))
        return super().predict(sample)