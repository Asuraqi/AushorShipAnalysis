import jieba
result_file = open(r'C:\Users\lenovo\Desktop\大数据技术\dict.txt', "r", encoding='utf-8')
lines=result_file.readlines()

result_file = open(r'C:\Users\lenovo\Desktop\大数据技术\function_word.txt', "w+", encoding='utf-8')
for line in lines:
    a,b,c=line.strip('\n').split(' ')
    print(c)
    if (c=='d' or c=='p'or c=='c'or c=='u'or c=='e'or c=='o')and int(b)>2000:
        print(a)
        result_file.write(a+'\t'+b+'\t'+c+'\n')
print(result_file.readline())
