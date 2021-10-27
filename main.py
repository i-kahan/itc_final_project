import pickle
import pandas as pd
import numpy as np
import features_generation
import pydub

MODEL_PATH = 'model.pkl'
ENCODER_PATH = 'encode_targets.pkl'

with open(MODEL_PATH, 'rb') as pkl:
    MODEL = pickle.load(pkl)

with open(ENCODER_PATH, 'rb') as pkl:
    ENCODER = pickle.load(pkl)


class GenreClassifier:

    def __init__(self):
        self.model_ = MODEL
        self.encoder_ = ENCODER

    @staticmethod
    def get_data(path):

        return features_generation.get_data(path)

    @staticmethod
    def generate_features(data, sr):
        return features_generation.get_all_features_from_data(data, sr)

    def predict(self, data):
        predict = self.model_.predict(data)
        return self.encoder_.inverse_transform(predict)

    def predict_from_path(self, path):
        data = features_generation.get_all_features_from_path(path)
        predict = self.model_.predict(data)
        return self.encoder_.inverse_transform(predict)


def main():
    clf = GenreClassifier()
    x, sr, file_name = clf.get_data('C:\Download\jazz.00004.wav')
    features = clf.generate_features(x, sr)
    p = clf.predict([features])
    print(p)


if __name__ == '__main__':
    main()
