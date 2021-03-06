import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize


def TFIDF(X_train, X_test, MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    print("tf-idf with", str(np.array(X_train).shape[1]), "features")
    return (X_train, X_test)


if __name__ == '__main__':
    train_pd = pd.read_csv('./frac=0.8/training_set_0.8.csv')
    test_pd = pd.read_csv('./frac=0.8/testing_set_0.8.csv')

    X_train = train_pd['text'].tolist()
    X_test = test_pd['text'].tolist()

    tfidf_train, tfidf_test = TFIDF(X_train, X_test)

    print(tfidf_train)