import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier


FILE_PATH = 'GTZAN/features_30_sec.csv'
MODEL_FILE = 'model.pkl'


def train(x, y):
    model = RandomForestClassifier()
    model.fit(x, y)
    return model


def main():
    df = pd.read_csv(FILE_PATH)
    x = df.drop(columns = ["filename", 'label'])
    y = df['label']
    model = train(x, y)
    pickle.dump(model, open(MODEL_FILE, 'wb'))


if __name__ == '__main__':
    main()
