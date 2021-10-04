import numpy as np
import librosa
# import IPython.display as ipd
# import matplotlib.pyplot as plt
# import librosa.display
import os


def get_data(path):
    directory, file_name = os.path.split(path)
    x, sr = librosa.load(path)
    return x, sr, file_name


def get_all_features_from_data(data, sr):

    length = data.shape[0]

    chroma_stft_mean = np.mean(librosa.feature.chroma_stft(y=data, sr=sr))
    chroma_stft_var = np.var(librosa.feature.chroma_stft(y=data, sr=sr))

    rms_mean = np.mean(librosa.feature.rms(y=data))
    rms_var = np.var(librosa.feature.rms(y=data))

    spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=data, sr=sr))
    spectral_centroid_var = np.var(librosa.feature.spectral_centroid(y=data, sr=sr))

    spectral_bandwidth_mean = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sr))
    spectral_bandwidth_var = np.var(librosa.feature.spectral_bandwidth(y=data, sr=sr))

    rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sr))
    rolloff_var = np.var(librosa.feature.spectral_rolloff(y=data, sr=sr))

    zero_crossing_rate_mean = np.mean(librosa.feature.zero_crossing_rate(y=data))
    zero_crossing_rate_var = np.var(librosa.feature.zero_crossing_rate(y=data))

    harmony_mean = np.mean(librosa.effects.harmonic(y=data))
    harmony_var = np.var(librosa.effects.harmonic(y=data))

    perceptr_mean = np.mean(librosa.effects.percussive(y=data))
    perceptr_var = np.var(librosa.effects.percussive(y=data))

    tempo = librosa.beat.beat_track(y=data, sr=sr)[0]

    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20)

    mfcc1_mean = np.mean(mfcc[0])
    mfcc1_var = np.var(mfcc[0])

    mfcc2_mean = np.mean(mfcc[1])
    mfcc2_var = np.var(mfcc[1])

    mfcc3_mean = np.mean(mfcc[2])
    mfcc3_var = np.var(mfcc[2])

    mfcc4_mean = np.mean(mfcc[3])
    mfcc4_var = np.var(mfcc[3])

    mfcc5_mean = np.mean(mfcc[4])
    mfcc5_var = np.var(mfcc[4])

    mfcc6_mean = np.mean(mfcc[5])
    mfcc6_var = np.var(mfcc[5])

    mfcc7_mean = np.mean(mfcc[6])
    mfcc7_var = np.var(mfcc[6])

    mfcc8_mean = np.mean(mfcc[7])
    mfcc8_var = np.var(mfcc[7])

    mfcc9_mean = np.mean(mfcc[8])
    mfcc9_var = np.var(mfcc[8])

    mfcc10_mean = np.mean(mfcc[9])
    mfcc10_var = np.var(mfcc[9])

    mfcc11_mean = np.mean(mfcc[10])
    mfcc11_var = np.var(mfcc[10])

    mfcc12_mean = np.mean(mfcc[11])
    mfcc12_var = np.var(mfcc[11])

    mfcc13_mean = np.mean(mfcc[12])
    mfcc13_var = np.var(mfcc[12])

    mfcc14_mean = np.mean(mfcc[13])
    mfcc14_var = np.var(mfcc[13])

    mfcc15_mean = np.mean(mfcc[14])
    mfcc15_var = np.var(mfcc[14])

    mfcc16_mean = np.mean(mfcc[15])
    mfcc16_var = np.var(mfcc[15])

    mfcc17_mean = np.mean(mfcc[16])
    mfcc17_var = np.var(mfcc[16])

    mfcc18_mean = np.mean(mfcc[17])
    mfcc18_var = np.var(mfcc[17])

    mfcc19_mean = np.mean(mfcc[18])
    mfcc19_var = np.var(mfcc[18])

    mfcc20_mean = np.mean(mfcc[19])
    mfcc20_var = np.var(mfcc[19])

    features = [length, chroma_stft_mean, chroma_stft_var, rms_mean,
                rms_var, spectral_centroid_mean, spectral_centroid_var,
                spectral_bandwidth_mean, spectral_bandwidth_var, rolloff_mean,
                rolloff_var, zero_crossing_rate_mean, zero_crossing_rate_var,
                harmony_mean, harmony_var, perceptr_mean, perceptr_var, tempo,
                mfcc1_mean, mfcc1_var, mfcc2_mean, mfcc2_var, mfcc3_mean,
                mfcc3_var, mfcc4_mean, mfcc4_var, mfcc5_mean, mfcc5_var,
                mfcc6_mean, mfcc6_var, mfcc7_mean, mfcc7_var, mfcc8_mean,
                mfcc8_var, mfcc9_mean, mfcc9_var, mfcc10_mean, mfcc10_var,
                mfcc11_mean, mfcc11_var, mfcc12_mean, mfcc12_var, mfcc13_mean,
                mfcc13_var, mfcc14_mean, mfcc14_var, mfcc15_mean, mfcc15_var,
                mfcc16_mean, mfcc16_var, mfcc17_mean, mfcc17_var, mfcc18_mean,
                mfcc18_var, mfcc19_mean, mfcc19_var, mfcc20_mean, mfcc20_var]
    # print(features)
    return features


def get_all_features_from_path(path):

    data, sr, file_name = get_data(path)

    features = get_all_features_from_data(data, sr)

    return [file_name] + features
