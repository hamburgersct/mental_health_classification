# coding=gbk
import os
import csv
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from autocorrect import spell

path = "./passages/"
directories = os.listdir(path)


# print(directories)

# # rename csv files as student numbers
# for dir in directories:
#     full_dir = os.path.join(path,dir)
#     if os.path.isdir(full_dir):
#         files = os.listdir(full_dir)
#         for filename in files:
#             stu_num = filename[:9] + '.csv'
#             os.rename(os.path.join(full_dir,filename),os.path.join(full_dir,stu_num))

def Text_cleaner(text):
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''},  # remove spaces at the beginning
        {r'[^\w\s]': u' '}  #remove punctuation
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
    text = text.rstrip()
    return text.lower()


def Preprocessing(src_top_dir, dist_top_dir):
    '''
    :param top_dir: The directory of texts. in this project, it is './passage'
    :return:
    '''
    stop_words = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()
    word_count = []

    for dir in os.listdir(src_top_dir):
        dir_path = os.path.join(src_top_dir, dir)
        filenames = os.listdir(dir_path)
        for file in filenames:
            with open(os.path.join(dir_path, file), encoding='utf-8-sig') as csvFile:
                reader = csv.reader(csvFile)
                passage_list = []
                for row in reader:
                    passage_list += row
                passage = ' '.join(passage_list)

                # preprocessing
                passage_clean = Text_cleaner(passage)
                tokens = word_tokenize(passage_clean)
                filtered_sentence = [w for w in tokens if not w in stop_words]
                # TODO:lemmatization效果不好，可能需要进行词性标注
                lem_sentence = [wnl.lemmatize(w) for w in filtered_sentence]
                word_count.append(len(lem_sentence))
                dist_passage = ' '.join(lem_sentence)
            # write into .txt files
            dist_dir = os.path.join(dist_top_dir, dir)
            if not os.path.exists(dist_dir):
                os.makedirs(dist_dir)
            with open(os.path.join(dist_dir, file[:-4] + '.txt'), 'w') as txtFile:
                txtFile.write(dist_passage)
                txtFile.close()
            csvFile.close()
    # word count statistic features
    print('min:' + str(min(np.array(word_count))))
    print('max:' + str(max(np.array(word_count))))
    print('mean:' + str(np.mean(np.array(word_count))))

def Merge():
    stud_mental = pd.read_csv('./features_of_student.csv', encoding='gbk')
    stud_mental[['学号']] = stud_mental[['学号']].astype(int).astype(str)

    text_folder = './text_docs/'
    files = os.listdir(text_folder)
    texts = []
    student_numbers = []
    for file in files:
        file_path = os.path.join(text_folder, file)
        f = open(file_path)
        text = f.readline()
        student_numbers.append(file[:-4])
        texts.append(text)
    text_dict = {'学号': student_numbers,
                 'text': texts}
    text_pd = pd.DataFrame(text_dict)

    # merge two dataframes
    # 合并后只剩18级，因为17级没有心理数据
    data = pd.merge(stud_mental, text_pd)
    data.to_csv('student_data.csv', index=False)

def Split(sample_frac):
    data = pd.read_csv('./student_data.csv', encoding='utf-8')
    train_set = data.sample(frac=sample_frac, replace=False, random_state=1)
    test_set = data[~data.index.isin(train_set.index)]
    folder = 'frac=' + str(sample_frac)
    if folder not in os.listdir():
        os.mkdir(folder)
        if len(os.listdir(folder)) == 0:
            train_set.to_csv(os.path.join(folder, 'training_set_' + str(sample_frac) + '.csv'), index=False)
            test_set.to_csv(os.path.join(folder, 'testing_set_' + str(sample_frac) + '.csv'), index=False)

if __name__ == '__main__':
    # preprocessing('./passages', './passages_processed')
    # Merge()
    Split(0.8)
