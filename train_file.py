import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


FILE_PATH = 'GTZAN/features_30_sec.csv'
MODEL_FILE = 'model.pkl'
ENCODER_FILE = 'encode_targets.pkl'


def train(x, y):
    model = RandomForestClassifier()
    model.fit(x, y)
    return model


def encode_targets(y):
    le = LabelEncoder()
    le.fit(y)
    pickle.dump(le, open(ENCODER_FILE, 'wb'))
    return le.transform(y)


def main():
    df = pd.read_csv(FILE_PATH)
    x = df.drop(columns=["filename", 'label'])
    y = encode_targets(df['label'])
    model = train(x, y)
    pickle.dump(model, open(MODEL_FILE, 'wb'))


if __name__ == '__main__':
    main()
