# coding=gbk
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

def standard_svm():
    text_train = pd.read_csv('./frac=0.8/training_set_0.8.csv')
    text_test = pd.read_csv('./frac=0.8/testing_set_0.8.csv')

    X_train = text_train['text'].tolist()
    X_test = text_test['text'].tolist()
    y_train = text_train['可能问题'].tolist()
    y_test = text_test['可能问题'].tolist()

    pipeline = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier()),
                         ])
    parameters = {
        'vect__ngram_range': [(1,2), (1,3)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-4, 1e-6)
    }

    gs_clf = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1)
    gs_clf.fit(X_train, y_train)

    predicted = gs_clf.predict(X_test)

    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test, predicted))
    print(metrics.f1_score(y_test, predicted, average='weighted'))

def metadata_svm():
    text_train = pd.read_csv('./frac=0.8/training_set_0.8.csv')
    text_test = pd.read_csv('./frac=0.8/testing_set_0.8.csv')

    X_train = text_train['text'].tolist()
    X_test = text_test['text'].tolist()
    y_train = text_train['可能问题'].tolist()
    y_test = text_test['可能问题'].tolist()

    vectorizer_x = TfidfVectorizer()
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    print("tf-idf with", str(np.array(X_train).shape))

    metadata_train = np.array(text_train.iloc[:, 26:-2])
    metadata_test = np.array(text_test.iloc[:, 26:-2])

    print(np.array(X_train).shape, metadata_train.shape)

    combine_train = np.concatenate((X_train, metadata_train), axis=1)
    combine_test = np.concatenate((X_test, metadata_test), axis=1)

    pipeline = Pipeline([('clf', SGDClassifier())])
    parameters = {
        'clf__alpha': (1e-4, 1e-6)
    }

    gs_clf = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1)
    gs_clf.fit(combine_train, y_train)

    predicted = gs_clf.predict(combine_test)

    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test, predicted))
    print("F1 score: ", str(metrics.f1_score(y_test, predicted, average='weighted')))


if __name__ == '__main__':
    metadata_svm()
