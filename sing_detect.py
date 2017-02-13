import theano
import wave
import array
import numpy
import glob
import os
import os.path
import librosa
from python_speech_features import mfcc, fbank
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional
from keras.layers import Activation, Permute, TimeDistributed, Flatten, Convolution1D, Reshape, Convolution2D, MaxPooling2D
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, RMSprop
from keras.datasets import imdb

# For reproducity
numpy.random.seed(11901)


def get_feature_list(folder):
    return glob.glob(folder + '/*.npz')


def prepair_feature(feature_file):
    data = numpy.load(feature_file)
    audio = data['audio']
    annotation = data['annotation']

    audio = numpy.reshape(audio, [audio.shape[0], 1, audio.shape[1]])
    audio = numpy.log(audio)

    audio_sum = numpy.sum(audio, 2)
    audio = audio - audio_sum[:, :, numpy.newaxis] / audio.shape[2]
    audio_sum = numpy.sqrt(numpy.sum(audio ** 2, 2))
    audio /= audio_sum[:, :, numpy.newaxis]
    pad = 32 - audio.shape[0] % 32

    audio = numpy.concatenate((audio, numpy.zeros((pad, 1, audio.shape[2]))), 0)
    annotation = numpy.concatenate((annotation, numpy.zeros(pad)))

    return audio, annotation

def pop_layer(model):
    # if not model.outputs:
    #     raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

def train_model(model, epoch):
    for i in range(0, epoch):
        for feature_file in get_feature_list('Train'):
            print('Training use ' + feature_file)
            audio, annotation = prepair_feature(feature_file)
            model.reset_states()
            model.fit(audio, annotation, batch_size=32, nb_epoch=1, verbose=1)
        for feature_file in get_feature_list('Test'):
            print('Evaluating use ' + feature_file)
            audio, annotation = prepair_feature(feature_file)
            model.reset_states()
            a, b = model.evaluate(audio, annotation, batch_size=32, verbose=1)
            print(a)
            print(b)
            print('Saving to ' + str(i) + '.model')
        model.save(str(i) + '.model')

FEATURE_DIM = 4960

model = Sequential()
model.add(Dropout(0.6, batch_input_shape=(32, 1, FEATURE_DIM)))
model.add(TimeDistributed(Dense(256, input_dim=FEATURE_DIM, activation="relu"), batch_input_shape=(32, 1, FEATURE_DIM)))
model.add(Dropout(0.5))
model.add(
    Bidirectional(
        LSTM(256, return_sequences=True, stateful=True),
        batch_input_shape=(32, 1, FEATURE_DIM), merge_mode='sum'
    )
)

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.summary()
train_model(model, 16)

exit(0)
