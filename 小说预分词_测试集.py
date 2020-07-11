import os

import jieba.posseg

import config

DATA_DIR = config.TEST_DATA_DIR
SEG_DIR = config.TEST_SEG_DIR


for filename in os.listdir(DATA_DIR):
    # 训练数据文件
    data_path = os.path.join(DATA_DIR, filename)
    # 目标分词文件
    seg_file = filename + '.seg'
    seg_path = os.path.join(SEG_DIR, seg_file)

    author_id = eval(filename[:filename.find('-')])

    # 如果不是想要的作者，则跳过
    if author_id not in config.target_author:
        continue

    # 如果已经分词，则跳过
    if os.path.exists(seg_path):
        print(seg_path, '已存在')
        continue

    print('[DEBUG] =============> start ', filename)
    with open(seg_path, 'w', encoding='utf-8') as out:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                author_id, novel_id, chapter_id, content = line.strip('\n').split('\t')
                word_list = []
                for word, flag in jieba.posseg.cut(content):
                    if flag.startswith("x"):
                        continue
                    word_list.append(word + "/" + flag)
                out.write(' '.join(word_list) + '\n')
    print('[DEBUG] =============> done  ', filename)