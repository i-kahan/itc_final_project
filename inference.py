import numpy as np
import pandas as pd
import pickle
from flask import Flask, request
import os

import features_generation

src = "Bach Toccata and Fugue in D minor organ.mp3"
dst = 'BTF.wav'
MODEL_PATH = 'model.pkl'

with open(MODEL_PATH, 'rb') as pkl:
    MODEL = pickle.load(pkl)

app = Flask(__name__)


@app.route('/find_genre')
def find_genre():
    data, sr = request.args['data'], request.args['sr']
    features = np.array(features_generation.get_all_features_from_data(data, sr))
    return str(MODEL.predict(features)[0])


def main():
    port = int(os.environ.get('PORT'))
    app.run(host='0.0.0.0', port=port)


if __name__ == '__main__':
    main()
