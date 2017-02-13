import glob
import numpy
from PIL import Image, ImageDraw
from scipy import signal
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional
from keras.layers import Activation, Permute, TimeDistributed, Flatten, Convolution1D, Reshape, Convolution2D, MaxPooling2D
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, RMSprop
from keras.datasets import imdb

def prepare_feature(feature_file):
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

def get_feature_list(folder):
    return glob.glob(folder + '/*.npz')


def get_audio_list(folder):
    audio_list = glob.glob(folder + '/*.wav')
    annotation_list = []
    for audio_name in audio_list:
        name = os.path.splitext(os.path.basename(audio_name))[0]
        annotation_name = 'Annotate/' + name + '.lab'
        annotation_list.append(annotation_name)
    return audio_list, annotation_list


model = load_model('model_save.model')
for feature_file in get_feature_list('Test'):
    audio, annotation = prepare_feature(feature_file)
    model.reset_states()
    result = model.predict(audio)
    annotation = annotation.flatten()
    result = result.flatten()
    for i in range(0, len(result)):
        if result[i] > 0.5:
            result[i] = 1.0
        else:
            result[i] = 0.0
        if annotation[i] > 0.5:
            annotation[i] = 1.0
        else:
            annotation[i] = 0.0
    correct = 0
    wrong = 0
    for i in range(0, len(result)):
        if annotation[i] == result[i]:
            correct += 1
        else:
            wrong += 1
    print(correct / (correct + wrong))
    window = signal.gaussian(31, std=11)
    window /= window.sum()
    smoothed = numpy.convolve(result, window, mode='same')
    for i in range(0, len(result)):
        if smoothed[i] > 0.5:
            smoothed[i] = 1.0
        else:
            smoothed[i] = 0.0
    correct = 0
    wrong = 0
    print(sum(annotation))
    for i in range(0, len(result)):
        if annotation[i] == smoothed[i]:
            correct += 1
        else:
            wrong += 1

    length = len(result)
    im = Image.new('RGBA', (length, 590), (255, 255, 255, 0))
    draw = ImageDraw.Draw(im)

    for i in range(length):
        if annotation[i] == 1:
            draw.line((i, 0, i, 195), fill=0)
    for i in range(length):
        if result[i] == 1:
            draw.line((i, 200, i, 395), fill=0)
    for i in range(length):
        if smoothed[i] == 1:
            draw.line((i, 400, i, 595), fill=0)
    im.show()
    im.save(feature_file + ".bmp")
    #numpy.savez(feature_file + 'result.npz', result=result, annotation=annotation, smoothed=smoothed)

    print(correct / (correct + wrong))
    print(result.shape)
    print(annotation.shape)
