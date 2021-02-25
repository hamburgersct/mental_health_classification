import os
import csv
import re
import numpy as np
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

def text_cleaner(text):
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


def preprocessing(src_top_dir, dist_top_dir):
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
                passage_clean = text_cleaner(passage)
                tokens = word_tokenize(passage_clean)
                filtered_sentence = [w for w in tokens if not w in stop_words]
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


if __name__ == '__main__':
    preprocessing('./passages', './passages_processed')

