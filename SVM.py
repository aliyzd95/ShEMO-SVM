seed_value = 42
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value)
import numpy as np

np.random.seed(seed_value)
import tensorflow as tf

tf.random.set_seed(seed_value)

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsOneClassifier
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
import warnings
from opensmile_preprocessing import opensmile_Functionals, emo_labels

warnings.filterwarnings("ignore")


def generate_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return cnf_matrix


def plot_confusion_matrix(predicted_labels_list, y_test_list):
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision=2)
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=emo_labels, normalize=True, title='SVM + eGeMAPS')
    plt.show()


def svm(X, y):
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value)
    cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed_value)
    model = SVC()
    ovo = OneVsOneClassifier(model)
    space = dict()
    space['estimator__C'] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    space['estimator__gamma'] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    search = BayesSearchCV(ovo, space, scoring='recall_macro', cv=cv_inner, n_jobs=-1, verbose=0)
    pipeline = make_pipeline(StandardScaler(), search)
    scores = cross_validate(pipeline, X, y, scoring=['recall_macro', 'accuracy'], cv=cv_outer, n_jobs=-1, verbose=2)
    print('____________________ Support Vector Machine ____________________')
    print(f"Weighted Accuracy: {np.mean(scores['test_accuracy'] * 100)}")
    print(f"Unweighted Accuracy: {np.mean(scores['test_recall_macro']) * 100}")


X, y = opensmile_Functionals()

N_SAMPLES = X.shape[0]

perm = np.random.permutation(N_SAMPLES)
X = X[perm]
y = y[perm]

if __name__ == '__main__':
    svm(X, y)

    # Accuracy: 72.95645139237044
    # UAR: 58.66650383394545
