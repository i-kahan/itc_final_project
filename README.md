# itc_final_project

This project is the final submission of the ITC data science course.
It contains API for classifying the genre of songs based on the GTZAN database.

## Installation
In order to use this API, one must have the following modules:

numpy<=1.20
librosa
os
pydub
pickle
pandas
sklearn
tensorflow

## Usage

There are 2 important file for this project, main and features_generation. the first holds the API class, and the second is a helper to create the features 
from the raw data.
The models of the project are prepared to use in 2 ways:
1. download the pkl file contains them.
2. Run the train_file to create new pkl files.

One can call this API by using this line:
clf = GenreClassifier()

This class contains 4 methods:
1. GenreClassifier.get_data(path) - get the data from the file with 'path'
2. GenreClassifier.generate_features(data) - after getting the data creating the features for the models.
3. GenreClassifier.predict(features) - predicting the genre with the features.
4. GenreClassifier.predict_from_path - gathering all 3 steps above in 1 method, from file path to classification
 
 ## credits
Aviad Haviv
Ilan Kahan
Tovi Benoni
Yehuda Shvut
