import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import pymysql

MYSQL_HOST = 'localhost'
MYSQL_DB = 'message'
MYSQL_USER = 'root'
MYSQL_PASS = '123'

connection = pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER,
                             password=MYSQL_PASS, db=MYSQL_DB,
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

cursor = connection.cursor()

sql = 'SELECT message FROM message'


cursor.execute(sql)
item_list = cursor.fetchall()
message_list = [r['message'] for r in item_list]

def wordcloudplot(data, file_name):
    wordcloud = WordCloud(font_path='simhei.ttf',
                          background_color='white',
                          margin=3,
                          max_words=100,
                          max_font_size=50,
                          random_state=42)
    wordcloud = wordcloud.generate(data)
    wordcloud.to_file(file_name)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


import jieba1.posseg
def display_wordcloud(words_list, file_name):
    raw_text = ' '.join(words_list)

    words = []

    cnt = 0
    for word, flag in jieba1.posseg.cut(raw_text):
        if len(word) >1 and flag.startswith('n'):
            words.append(word)
        cnt += 1
        if cnt >10000:
            break
    wordcloudplot(' '.join(words), file_name)


display_wordcloud(message_list, 'D:\\1.jpg')
