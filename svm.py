# coding=gbk
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer

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
        'clf__alpha': (1e-4, 1e-6)
    }

    # Training config
    kfold = StratifiedKFold(n_splits=5)
    scoring = {'Accuracy': 'accuracy', 'F1': 'f1_macro'}
    refit = 'F1'

    gs_clf = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1, cv=kfold, scoring=scoring, refit=refit)
    gs_clf.fit(X_train, y_train)

    predicted = gs_clf.predict(X_test)

    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test, predicted))
    print(metrics.f1_score(y_test, predicted, average='weighted'))


# Create Function Transformer to use Feature Union
def get_numeric_data(x):
    return np.array(x.iloc[:, 26:-2])

def get_text_data(x):
    return x['text'].tolist()

def metadata_svm_fu():
    text_train = pd.read_csv('./frac=0.8/training_set_0.8.csv')
    text_test = pd.read_csv('./frac=0.8/testing_set_0.8.csv')

    y_train = text_train['可能问题'].tolist()
    y_test = text_test['可能问题'].tolist()

    transformer_numberic = FunctionTransformer(get_numeric_data)
    transformer_text = FunctionTransformer(get_text_data)

    # Create a pipeline to concatenate Tfidf Vector and Numeric data
    # Use SVM as classifier
    pipeline = Pipeline([
        ('metadata', FeatureUnion([
            ('numeric_feature', Pipeline([
                ('selector', transformer_numberic)
            ])),
            ('text_features', Pipeline([
                ('selector', transformer_text),
                ('vec', TfidfVectorizer(ngram_range=(1,3)))
            ]))
        ])),
        ('clf', SGDClassifier())
    ])

    # Grid Search Parameters for SGDClassifer
    parameters = {
        'clf__alpha': (1e-4, 1e-6)
    }

    # Training config
    kfold = StratifiedKFold(n_splits=5)
    scoring = {'Accuracy': 'accuracy', 'F1': 'f1_macro'}
    refit = 'F1'

    gs_clf = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1, cv=kfold, scoring=scoring, refit=refit)
    gs_clf.fit(text_train, y_train)

    predicted = gs_clf.predict(text_test)

    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test, predicted))
    print("F1 score: ", str(metrics.f1_score(y_test, predicted, average='weighted')))


if __name__ == '__main__':
    # metadata_svm_fu()
    # standard_svm()
    oversamp_svm_fu()