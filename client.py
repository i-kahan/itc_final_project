import numpy as np
import pandas as pd
import requests
from sys import argv
import os
import librosa
import pydub

REQUEST_URL = 'https://fininal-project-app.herokuapp.com/find_genre'


def change2wav(src):
    sound = pydub.AudioSegment.from_mp3(src)
    return sound


def main(argv):
    directory, file_name = os.path.split(argv[1])
    if file_name.endswith('.mp3'):
        x, sr = change2wav(librosa.load(argv[1]))
    else:
        x, sr = librosa.load(argv[1])
    d = {'data': x, 'sr': sr}
    genre = requests.get(REQUEST_URL, params=d).text
    print(genre)


if __name__ == '__main__':
    main(argv)
