seed_value = 42
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value)
import numpy as np

np.random.seed(seed_value)
import tensorflow as tf

tf.random.set_seed(seed_value)

import opensmile

path = 'shemo'  # download and extract files from https://github.com/aliyzd95/ShEMO-Modification/raw/main/shemo.zip
emo_codes = {"A": 0, "W": 1, "H": 2, "S": 3, "N": 4, "F": 5}
emo_labels = ["anger", "surprise", "happiness", "sadness", "neutral", "fear"]


def get_emotion_label(file_name):
    emo_code = file_name[3]
    return emo_codes[emo_code]


def opensmile_Functionals():
    feature_extractor = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        verbose=True, num_workers=None,
        sampling_rate=16000, resample=True,
    )
    features = []
    emotions = []
    for file in os.listdir(path):
        if emo_labels[get_emotion_label(file)] != 'fear':
            df = feature_extractor.process_file(f'{path}/{file}')
            features.append(df)
            emotions.append(get_emotion_label(file))
    features = np.array(features).squeeze()
    emotions = np.array(emotions)
    return features, emotions
