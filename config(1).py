# 目标作者ID
target_author = target_author = [998, 643, 626, 4, 1002, 1000, 1709, 645, 735, 1035, 644, 2, 612, 1, 3, 746, 1010, 591,
                                 495, 1838, 580, 228, 999, 426, 979, 1747, 671, 627, 494, 1855, 1839, 973, 149, 387,
                                 1764, 3081, 1949, 55, 362, 988, 142, 150, 981, 172, 398, 926, 328, 251, 925, 352, 401,
                                 1869, 336, 2022, 155, 454, 407, 2040, 1040, 629, 54, 708, 405, 984, 895, 1849, 604,
                                 546, 1721, 53, 380, 117, 399, 52, 1005, 1749]
target_author = target_author[:5]
# target_author = [4, 2, 580]


# 原训练/测试 文件所在目录
TRAIN_DATA_DIR = r"C:\Users\Administrator\Desktop\telegram\小说\TrainData\origin"
TEST_DATA_DIR = r"C:\Users\Administrator\Desktop\telegram\小说\TestData"

# 训练/测试 分词文件
TRAIN_SEG_DIR = r"C:\Users\Administrator\Desktop\telegram\小说\预分词\train"
TEST_SEG_DIR = r"C:\Users\Administrator\Desktop\telegram\小说\预分词\test"

# 功能词文件路径
FUNCTION_WORD_PATH = r"C:\Users\Administrator\Desktop\project\pyproject\pytest\data\function_word_300.txt"

"""
    提取特征
"""

TITLE = [
    "作者ID",
    "段落数平均值", "段落平均长度", "句子平均长度",  # 3
    "问号占比", "句号占比", "逗号占比", "感叹号占比", "引号占比", "冒号占比", "省略号占比", "分号占比", "单括引号占比",  # 9
    "1字词占比", "2字词占比", "3字词占比", "4字词占比", "5字词占比", "6字词占比", "7字词占比", "8字词占比",  # 8
    "名词占比", "副词占比", "形容词占比", "介词占比", "连词占比", "助词占比", "叹词占比", "动词占比", "拟声词占比"  # 9
]

# 问号数,句号数,逗号数,感叹号数地,引号数,冒号数,省略号数（三个点）,分号数，单括引号（「和」）
PUNCT_CHAR_LIST = [
    ['?', '？'],
    ['.', '。'],
    [',', '，'],
    ['!', '！'],
    ['"', '“', '”'],
    [':', '：'],
    ['…'],
    [';', '；'],
    ['「', '」']
]
