import theano
import wave
import array
import numpy
import glob
import os
import os.path
import librosa
from python_speech_features import mfcc, fbank

numpy.random.seed(11901)

def get_audio_list(folder):
    audio_list = glob.glob(folder + '/*.wav')
    annotation_list = []
    for audio_name in audio_list:
        name = os.path.splitext(os.path.basename(audio_name))[0]
        annotation_name = 'Annotate/' + name + '.lab'
        annotation_list.append(annotation_name)
    return audio_list, annotation_list


def audio_to_matrix(music, annotation):
    music = wave.open(music, 'rb')
    frame_rate = music.getframerate()
    frame_count = music.getnframes()
    channel_count = music.getnchannels()
    music_data_raw = music.readframes(frame_count)
    music_data = array.array("h")
    music_data.fromstring(music_data_raw)
    mixed = music_data

    SEGMENT_SIZE = frame_rate
    segment_count = len(mixed) // SEGMENT_SIZE

    print(segment_count)
    mfcc_data = None
    for i in range(segment_count):
        sub_audio = numpy.array(mixed[i * SEGMENT_SIZE:(i + 1) * SEGMENT_SIZE])
        sub_data, energy = fbank(
            sub_audio, samplerate=frame_rate,
            nfilt=80, winlen=0.032, winstep=0.016, highfreq=8000, nfft=2048
        )
        sub_data = sub_data.flatten()
        sub_data = sub_data[numpy.newaxis, :]
        if(i % 100 == 0):
            print(i)
        if(mfcc_data is None):
            mfcc_data = sub_data
        else:
            mfcc_data = numpy.concatenate((mfcc_data, sub_data))
    annotation_data = numpy.zeros(mfcc_data.shape[0])
    print(mfcc_data.shape)
    annotation_file = open(annotation, "r")
    for line in annotation_file:
        if len(line) <= 0:
            continue
        line_parts = line.split()
        if len(line_parts) < 3 or line_parts[2] != 'sing':
            continue
        begin = float(line_parts[0])
        end = float(line_parts[1])
        begin_mfcc = round(begin)
        end_mfcc = round(end)
        if end_mfcc < begin_mfcc:
            continue
        annotation_data[begin_mfcc:end_mfcc] = 1
    print(mfcc_data.shape)
    print(annotation_data.shape)
    return mfcc_data, annotation_data


def prepare_data(folder):
    audio_list, annotation_list = get_audio_list(folder)
    print(audio_list)
    print(annotation_list)
    full_audio = None
    full_annotation = None

    for audio, annotation in zip(audio_list, annotation_list):
        feature_file = os.path.splitext(audio)[0] + '.npz'
        print('preprocessing ' + audio + ' to ' + feature_file)
        audio_mat, annotation_mat = audio_to_matrix(audio, annotation)
        numpy.savez(
            feature_file,
            audio=audio_mat, annotation=annotation_mat
        )

prepare_data('Train')
prepare_data('Test')
