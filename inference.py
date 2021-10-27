import numpy as np
import pickle
from flask import Flask, request
import os

import features_generation

MODEL_PATH = 'model.pkl'
ENCODER_PATH = 'encode_targets.pkl'

with open(MODEL_PATH, 'rb') as pkl:
    MODEL = pickle.load(pkl)

with open(ENCODER_PATH, 'rb') as pkl:
    ENCODER = pickle.load(pkl)


app = Flask(__name__)


@app.route('/find_genre')
def find_genre():
    data, sr = request.args['data'], request.args['sr']
    features = np.array(features_generation.get_all_features_from_data(data, sr))
    predict = MODEL.predict(features)
    return ENCODER.inverse_transform(predict)


def main():
    port = int(os.environ.get('PORT'))
    app.run(host='0.0.0.0', port=port)


if __name__ == '__main__':
    main()
