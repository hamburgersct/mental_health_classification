import os
import csv

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



def preprocessing(src_top_dir,dist_top_dir):
    '''
    :param top_dir: The directory of texts. in this project, it is './passage'
    :return:
    '''
    for dir in os.listdir(src_top_dir):
        dir_path = os.path.join(src_top_dir,dir)
        filenames = os.listdir(dir_path)
        for file in filenames:
            with open(os.path.join(dir_path,file), encoding='utf-8-sig') as csvFile:
                reader = csv.reader(csvFile)
                passage_list = []
                for row in reader:
                    passage_list += row
                passage = ' '.join(passage_list)

                # TODO:preprocess texts

            dist_dir = os.path.join(dist_top_dir,dir)
            if not os.path.exists(dist_dir):
                os.makedirs(dist_dir)
            with open(os.path.join(dist_dir,file[:-4]+'.txt'), 'w') as txtFile:
                txtFile.write(passage)
                txtFile.close()
            csvFile.close()


if __name__ == '__main__':
    preprocessing('./passages','./passages_processed')



# TODO：analyse statistic characteristics

