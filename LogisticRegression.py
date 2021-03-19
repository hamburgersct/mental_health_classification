# coding=gbk
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import FunctionTransformer, LabelEncoder

columns = ['学号', '性别', '生源地', '总分', '幻觉、妄想症状', '自杀意图', '焦虑指标总分', '抑郁指标总分', '偏执指标总分', '自卑指标总分',
           '敏感指标总分', '社交恐惧指标总分', '躯体化指标总分', '依赖指标总分', '敌对攻击指标总分', '冲动指标总分', '强迫指标总分',
           '网络成瘾指标总分', '自伤行为指标总分', '进食问题指标总分', '睡眠困扰指标总分', '学校适应困难指标总分', '人际关系困扰指标总分',
           '学业压力指标总分', '就业压力指标总分', '恋爱困扰指标总分']

def standard_lr():
    data = pd.read_csv('student_data.csv')
    data.drop(columns=columns, inplace=True)

    train_df, test_df = train_test_split(data, test_size=0.2, random_state=11)

    X_train = train_df['text'].tolist()
    X_test = test_df['text'].tolist()
    y_train = train_df['可能问题'].tolist()
    y_test = test_df['可能问题'].tolist()

    pipeline = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         # ('dim_red', TruncatedSVD()),
                         ('clf', LogisticRegression()),
                         ])
    parameters = {
        'vect__ngram_range': [(1, 2), (1, 3)],
        'clf__C': [1e-2, 1e-1, 1e0, 1e1]
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
    print("precision: ", str(metrics.precision_score(y_test, predicted, average='weighted')))
    print("accuracy: ", str(metrics.accuracy_score(y_test, predicted)))
    print("F1 score: ", str(metrics.f1_score(y_test, predicted, average='weighted')))
    print("recall: ", str(metrics.recall_score(y_test, predicted, average='weighted')))


# Create Function Transformer to use Feature Union
def get_numeric_data(x):
    return np.array(x.iloc[:, 0:-2])


def get_text_data(x):
    return x['text'].tolist()


def metadata_lr_fu():
    # train_df = pd.read_csv('./frac=0.8/training_set_0.8.csv')
    # test_df = pd.read_csv('./frac=0.8/testing_set_0.8.csv')

    data = pd.read_csv('student_data.csv', encoding='utf-8')
    data.drop(columns=columns, inplace=True)

    train_df, test_df = train_test_split(data, test_size=0.2, random_state=8)

    y_train = train_df['可能问题'].tolist()
    y_test = test_df['可能问题'].tolist()

    transformer_numeric = FunctionTransformer(get_numeric_data)
    transformer_text = FunctionTransformer(get_text_data)

    # Create a pipeline to concatenate Tfidf Vector and Numeric data
    # Use SVM as classifier
    pipeline = Pipeline([
        ('metadata', FeatureUnion([
            ('numeric_feature', Pipeline([
                ('selector', transformer_numeric)
            ])),
            ('text_features', Pipeline([
                ('selector', transformer_text),
                ('vec', TfidfVectorizer())
            ]))
        ])),
        ('clf', LogisticRegression())
    ])

    # Grid Search Parameters for SGDClassifer
    parameters = {
        'clf__C': (1e-2, 1e-1, 1),
        'metadata__text_features__vec__ngram_range': [(1, 2), (1, 3)],
        'metadata__text_features__vec__use_idf': [True, False]
    }

    # Training config
    kfold = StratifiedKFold(n_splits=5)
    scoring = {'Accuracy': 'accuracy', 'F1': 'f1_macro'}
    refit = 'F1'

    gs_clf = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1, cv=kfold, scoring=scoring, refit=refit)
    gs_clf.fit(train_df, y_train)

    predicted = gs_clf.predict(test_df)

    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test, predicted))
    print("precision: ", str(metrics.precision_score(y_test, predicted, average='weighted')))
    print("accuracy: ", str(metrics.accuracy_score(y_test, predicted)))
    print("F1 score: ", str(metrics.f1_score(y_test, predicted, average='weighted')))
    print("recall: ", str(metrics.recall_score(y_test, predicted, average='weighted')))


# Apply SMOTE to oversample training set
def oversamp_lr_fu():

    # train_df = pd.read_csv('./frac=0.8/training_set_0.8.csv')
    # test_df = pd.read_csv('./frac=0.8/testing_set_0.8.csv')

    data = pd.read_csv('student_data.csv', encoding='utf-8')
    data.drop(columns=columns, inplace=True)

    train_df, test_df = train_test_split(data, test_size=0.2, random_state=8)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df['可能问题'])
    y_test = encoder.fit_transform(test_df['可能问题'])


    # train_df.drop(columns=columns, inplace=True)
    # test_df.drop(columns=columns, inplace=True)

    transformer_numberic = FunctionTransformer(get_numeric_data)
    transformer_text = FunctionTransformer(get_text_data)

    pipeline = Pipeline([
        ('metadata', FeatureUnion([
            ('numeric_feature', Pipeline([
                ('selector', transformer_numberic)
            ])),
            ('text_features', Pipeline([
                ('selector', transformer_text),
                ('vec', TfidfVectorizer())
            ]))
        ])),
        ('oversample', SMOTE(random_state=8)),
        # ('reduce_dim', TruncatedSVD()),
        ('clf', LogisticRegression(max_iter=200))
    ])

    # Grid Search Parameters for SGDClassifer
    parameters = {
        'clf__C': (1e-1, 1, 10),
        'metadata__text_features__vec__ngram_range': [(1, 2), (1, 3)],
        # 'metadata__text_features__vec__use_idf': [True, False]
    }

    # Training config
    kfold = StratifiedKFold(n_splits=5)
    scoring = {'Accuracy': 'accuracy', 'F1': 'f1_macro'}
    refit = 'F1'

    gs_clf = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1, cv=kfold, scoring=scoring, refit=refit)
    gs_clf.fit(train_df, y_train)

    predicted = gs_clf.predict(test_df)

    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test, predicted))
    print("precision: ", str(metrics.precision_score(y_test, predicted, average='weighted')))
    print("accuracy: ", str(metrics.accuracy_score(y_test, predicted)))
    print("F1 score: ", str(metrics.f1_score(y_test, predicted, average='weighted')))
    print("recall: ", str(metrics.recall_score(y_test, predicted, average='weighted')))



if __name__ == '__main__':
    # standard_lr()
    # metadata_lr_fu()
    oversamp_lr_fu()
