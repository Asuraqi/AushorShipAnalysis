import jieba
if __name__ == '__main__':
    # result_file = open(segmented_path, "w+", encoding='utf-8')
    files = get_files(segmented_path)
    for i in files:
        print(i.split('/')[-2][:-4])
        list1 = []
        c = open(i, encoding='utf-8')
        line=c.readline()
        while line:
            a = line.split('\t')
            b = a[3:4]  # 这是选取需要读取的位数
            txt = open(r'data/segmented/'+"".join(a[2:3])+'.txt',"w",encoding='utf-8')
            txt.write("".join(b))
            list1.append(b)  # 将其添加在列表之中
            line = c.readline()
        c.close()