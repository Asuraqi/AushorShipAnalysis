import os
import re
import numpy as np
import jieba
import jieba.posseg
import config

"""
    修改配置
"""
DATA_DIR = config.TEST_DATA_DIR
SEG_DIR = config.TEST_SEG_DIR
FEATURE_FILE = r"data\feature\test_feature_author_%s.csv" % len(config.target_author)




"""
    读取功能词  
        建立映射 word ==> 下标
"""
function_word_index_dict = {}
function_word_list = []
with open(config.FUNCTION_WORD_PATH, 'r', encoding='utf-8') as f:
    for i, line in enumerate([line.strip('\n') for line in f.readlines()]):
        word, _, flag = line.split('\t')
        function_word_index_dict[word] = i
        function_word_list.append([word, flag])
config.TITLE.extend(["(虚词)%s:%s:%s" % (i, word, flag) for i, (word, flag) in enumerate(function_word_list)])

"""
    已提取的作者
"""
author_set = set()
if os.path.exists(FEATURE_FILE):
    with open(FEATURE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            author_id = line[:line.find(',')]
            author_set.add(eval(author_id))
out = open(FEATURE_FILE, 'a+', encoding='utf-8')
# out.write(','.join(TITLE) + '\n')

for filename in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, filename)

    # 提取作者ID
    author_id = eval(filename[:filename.find('-')])

    # 如果不是想要的作者，则跳过
    if author_id not in config.target_author:
        continue

    # 如果已经提取，则跳过
    if author_id in author_set:
        print('=====>', author_id, '已经读取')
        continue

    print('[DEBUG] =============> start ', filename)
    """
        读取对应的预分词文件
    """
    seg_list = []
    with open(os.path.join(SEG_DIR, filename + '.seg'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            seg_list.append(line)

    with open(file_path, 'r', encoding='utf-8') as f:
        # 按章遍历
        line_list = f.readlines()
        for line_index, line in enumerate(line_list):
            out_list = []
            paragraph_num_list = []  # 每章段落数
            punct_num_list = [[] for i in range(9)]  # 每章标点数统计
            paragraph_length_list = []  # 所有段落长度
            sentence_length_list = []  # 所有句子长度
            token_length_list = [0 for i in range(100)]  # 所有分词长度
            wordtype_length_list = [0 for i in range(30)]  # 所有词性数量
            function_word_num_list = [0] * len(function_word_index_dict)  # 统计所有功能性数量
            author_id, novel_id, chapter_id, content = line.strip('\n').split('\t')
            # 【标签】
            out_list.append(eval(author_id))

            """
                段落统计
            """
            for para in content.split(' '):
                paragraph_length_list.append(len(para))
                sentence_content_list = re.split(r'？|。|！|……', para)  # 划分句子
                sentence_length_list.extend([len(s) for s in sentence_content_list if s])
            paragraph_num_list.append(content.count(' ') + 1)
            """
                段落特征
            """
            # 【特征】段落数平均值
            paragraph_num_mean = np.mean(paragraph_num_list)
            out_list.append(paragraph_num_mean)
            # 【特征】段落平均长度
            paragraph_length_mean = np.mean(paragraph_length_list)
            out_list.append(paragraph_length_mean)
            # 【特征】句子平均长度
            sentence_length_mean = np.mean(sentence_length_list)
            out_list.append(sentence_length_mean)

            """
                标点统计
            """
            for i in range(len(config.PUNCT_CHAR_LIST)):
                char_sum = np.sum([content.count(c) for c in config.PUNCT_CHAR_LIST[i]])
                punct_num_list[i].append(char_sum)
            """
                标点特征
            """
            punct_sum_list = [np.sum(i) for i in punct_num_list]
            # 【特征】各标点占文章标点数的比例
            punct_sum = np.sum(punct_sum_list)
            punct_rate_list = [0] * len(punct_sum_list)
            if punct_sum > 0:
                punct_rate_list = np.array(punct_sum_list) / punct_sum
            out_list.extend(punct_rate_list[0:9])

            """
                分词统计
            """
            for wf in seg_list[line_index].split(' '):
                word, flag = wf.split('/')
                # 过滤非语素词
                if flag.startswith("x"):
                    continue
                # 分词长度统计
                if len(word) > 20:
                    continue
                token_length_list[len(word)] += 1
                # 分词词性统计。名词、副词、形容词、介词、连词、助词、叹词、动词、拟声词
                # 先判断vd再判断v
                if flag.startswith("n"):
                    wordtype_length_list[0] += 1
                elif flag.startswith("d"):
                    wordtype_length_list[1] += 1
                elif flag.startswith("a"):
                    wordtype_length_list[2] += 1
                elif flag.startswith("p"):
                    wordtype_length_list[3] += 1
                elif flag.startswith("c"):
                    wordtype_length_list[4] += 1
                elif flag.startswith("u"):
                    wordtype_length_list[5] += 1
                elif flag.startswith("e"):
                    wordtype_length_list[6] += 1
                elif flag.startswith("v"):
                    wordtype_length_list[7] += 1
                elif flag.startswith("o"):
                    wordtype_length_list[8] += 1

                """
                    统计功能词
                """
                if word in function_word_index_dict:
                    index = function_word_index_dict[word]
                    _, flag2 = function_word_list[index]
                    # 词性必须一致。例如"所"在不同语境中，词性不一定相同
                    if flag == flag2:
                        function_word_num_list[index] += 1
            """
                分词特征
            """
            total_token = np.sum(token_length_list)  # 总分词数

            # 【特征】各长度分词 所占比例
            token_length_rate = np.array(token_length_list) / total_token
            out_list.extend(token_length_rate[1:9])

            # 【特征】各词性分记号 所占比例
            wordtype_length_rate = np.array(wordtype_length_list) / total_token
            out_list.extend(wordtype_length_rate[0:9])

            # 【特征】各功能词 所占比例
            function_word_rate = function_word_num_list.copy()

            for i, (word, flag) in enumerate(function_word_list):
                if function_word_num_list[i] == 0:
                    continue
                if flag.startswith('d'):
                    function_word_rate[i] /= wordtype_length_list[1]
                elif flag.startswith('p'):
                    function_word_rate[i] /= wordtype_length_list[3]
                elif flag.startswith('c'):
                    function_word_rate[i] /= wordtype_length_list[4]
                elif flag.startswith('u'):
                    function_word_rate[i] /= wordtype_length_list[5]
                elif flag.startswith('e'):
                    function_word_rate[i] /= wordtype_length_list[6]
                elif flag.startswith('o'):
                    function_word_rate[i] /= wordtype_length_list[8]

            out_list.extend(function_word_rate)
            out_list = np.around(out_list, decimals=6)
            out.write(','.join(map(str, out_list)) + '\n')
    out.flush()
    print('[DEBUG] =============> done  ', filename)
out.close()
