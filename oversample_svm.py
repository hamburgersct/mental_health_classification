# coding=gbk
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer, LabelEncoder


# Create Function Transformer to use Feature Union
def get_numeric_data(x):
    return np.array(x.iloc[:, 0:-2])

def get_text_data(x):
    return x['text'].tolist()

# Apply SMOTE to oversample training set
def oversamp_svm_fu():
    text_train = pd.read_csv('./frac=0.8/training_set_0.8.csv')
    text_test = pd.read_csv('./frac=0.8/testing_set_0.8.csv')

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(text_train['��������'])
    y_test = encoder.fit_transform(text_test['��������'])

    columns = ['ѧ��', '�Ա�', '��Դ��', '�ܷ�', '�þ�������֢״', '��ɱ��ͼ', '����ָ���ܷ�', '����ָ���ܷ�', 'ƫִָ���ܷ�', '�Ա�ָ���ܷ�',
               '����ָ���ܷ�', '�罻�־�ָ���ܷ�', '���廯ָ���ܷ�', '����ָ���ܷ�', '�жԹ���ָ���ܷ�', '�嶯ָ���ܷ�', 'ǿ��ָ���ܷ�',
               '������ָ���ܷ�', '������Ϊָ���ܷ�', '��ʳ����ָ���ܷ�', '˯������ָ���ܷ�', 'ѧУ��Ӧ����ָ���ܷ�', '�˼ʹ�ϵ����ָ���ܷ�',
               'ѧҵѹ��ָ���ܷ�', '��ҵѹ��ָ���ܷ�', '��������ָ���ܷ�']

    text_train.drop(columns=columns, inplace=True)
    text_test.drop(columns=columns, inplace=True)

    transformer_numberic = FunctionTransformer(get_numeric_data)
    transformer_text = FunctionTransformer(get_text_data)

    pipeline = Pipeline([
        ('metadata', FeatureUnion([
            ('numeric_feature', Pipeline([
                ('selector', transformer_numberic)
            ])),
            ('text_features', Pipeline([
                ('selector', transformer_text),
                ('vec', TfidfVectorizer(ngram_range=(1, 3)))
            ]))
        ])),
        ('oversample', SMOTE(random_state=11)),
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
    print("precision: ", str(metrics.precision_score(y_test, predicted, average='weighted')))
    print("accuracy: ", str(metrics.accuracy_score(y_test, predicted)))
    print("F1 score: ", str(metrics.f1_score(y_test, predicted, average='weighted')))
    print("recall: ", str(metrics.recall_score(y_test, predicted, average='weighted')))


if __name__ == '__main__':
    oversamp_svm_fu()
