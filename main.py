import pickle
import features_generation
from sys import argv
import os

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


def main(arg):
    if os.path.isdir(arg[1]):
        for file in os.listdir(arg[1]):
            print('file name: ', file)
            clf = GenreClassifier()
            x, sr, file_name = clf.get_data(arg[1]+'\\'+file)
            features = clf.generate_features(x, sr)
            p = clf.predict([features])[0]
            print('prediction: ', p, '\n')
    else:
        clf = GenreClassifier()
        x, sr, file_name = clf.get_data(arg[1])
        features = clf.generate_features(x, sr)
        p = clf.predict([features])[0]
        print(p)


if __name__ == '__main__':
    main(argv)
