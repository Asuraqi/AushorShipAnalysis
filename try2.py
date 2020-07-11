target_set = set(["1000-r-00000_train_5",
"1002-r-00000_train_3",
"1005-r-00000_train_1",
"1010-r-00000_train_14",
"1035-r-00000_train_17",
"1040-r-00000_train_4",
"117-r-00000_train_1",
"150-r-00000_train_7",
"172-r-00000_train_5",
"1721-r-00000_train_2",
"1838-r-00000_train_1",
"1839-r-00000_train_1",
"1849-r-00000_train_1",
"1869-r-00000_train_4",
"1949-r-00000_train_4",
"2022-r-00000_train_4",
"2040-r-00000_train_5",
"228-r-00000_train_19",
"3081-r-00000_train_3",
"328-r-00000_train_4",
"336-r-00000_train_4",
"352-r-00000_train_4",
"362-r-00000_train_4",
"380-r-00000_train_3",
"398-r-00000_train_4",
"399-r-00000_train_2",
"401-r-00000_train_10",
"405-r-00000_train_4",
"407-r-00000_train_4",
"454-r-00000_train_3",
"495-r-00000_train_15",
"53-r-00000_train_2",
"546-r-00000_train_1",
"604-r-00000_train_3",
"627-r-00000_train_3",
"629-r-00000_train_2",
"643-r-00000_train_4",
"644-r-00000_train_3",
"645-r-00000_train_3",
"671-r-00000_train_4",
"708-r-00000_train_1",
"746-r-00000_train_4",
"895-r-00000_train_3",
"925-r-00000_train_4",
"926-r-00000_train_8",
"973-r-00000_train_18",
"981-r-00000_train_8",
"984-r-00000_train_4",
"999-r-00000_train_3"])

import os
SEG_DIR = r"E:\项目与代码\AuthorShipAnalysis\词云\data\output_train"
OUT_DIR = r"F:\小说"
for filename in os.listdir(SEG_DIR):
    filepath = os.path.join(SEG_DIR, filename)
    outpath = os.path.join(OUT_DIR, filename)
    if os.path.exists(outpath):
        continue
    print('=' * 60, filename)
    if filename[:-4] in target_set:
        infile = open(filepath, 'r', encoding='utf-8')
        outfile = open(outpath, 'w', encoding='utf-8')
        outfile.writelines(infile.readlines())
        outfile.flush()
        infile.close()
        outfile.close()
