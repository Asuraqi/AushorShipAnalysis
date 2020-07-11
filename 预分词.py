import os
import re
import numpy as np
import jieba
import jieba.posseg

TRAIN_DIR = r"E:\TrainData"
OUT_DIR = r"D:\output_train"


for train_file in os.listdir(TRAIN_DIR):

    file_path = os.path.join(TRAIN_DIR, train_file)
    out_path = os.path.join(OUT_DIR, train_file + ".seg")
    if os.path.exists(out_path):
        print('eeeee')
        continue
    print(file_path)

    with open(out_path, 'w', encoding='utf-8') as out:

        with open(file_path, 'r', encoding='utf-8') as f:

            for line in f.readlines():
                author_id, novel_id, chapter_id, content = line.strip('\n').split('\t')

                word_list = []
                for word, flag in jieba.posseg.cut(content):
                    if flag.startswith("x"):
                        continue
                    word_list.append(word + "/" + flag)
                out.write(' '.join(word_list) + '\n')