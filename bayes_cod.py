f = open('S:/Download/ML/news_train(2).txt', 'r', encoding='utf-8')
articles = f.readlines()
number_of_articles = len(articles)

print("start")

all_articles = []

for a in articles:
    a = a.lower()
    all_articles.append(a)
    
print("нижнее преобразование")
symbols = ['.',',','1','2','3','4','5','6','7','8','9','0','-','—',':','(',')','«','»']
articles = []

for a in all_articles:
    for symbol in symbols: 
        a = a.replace(str(symbol),'')
    articles.append(a)
 
print("преобразование символов")    

# science, style, culture, life, economics, business, travel, forces, media, sport
header_list = ['science', 'style', 'culture', 'life', 'economics', 'business', 'travel', 'forces', 'media', 'sport']
headers = {h: 0 for h in header_list}

from nltk.stem.snowball import RussianStemmer

stemmer = RussianStemmer(False)
articles_stemm = []
art = []

#articles[0] = articles[0][1:]

print("старт стемминга")

for article in articles:
    words_article = article.split()
    head = words_article.pop(0)
    headers[head] = headers[head] + 1
    
    for w in words_article:
        w = stemmer.stem(w)
        art.append(w)
        
    articles_stemm.append(art)
    art = []
  
print("конец стемминга")    

all_words_stemm = []

for a in articles_stemm:
    for w in a:
        all_words_stemm.append(w)
        
unique_words_stemm = list(set(all_words_stemm))

number_of_words = len(all_words_stemm)
number_of_unique_words = len(unique_words_stemm)

print(number_of_words)
print(number_of_unique_words)


count_of_words = {n:0 for n in unique_words_stemm}

for word in all_words_stemm:
    count_of_words[word] = count_of_words[word] + 1

min_count = 5

#for word in unique_words_stemm:
#    p_word[word] = count_of_words[word] / number_of_words

for word in unique_words_stemm:
    if (count_of_words[word] < min_count):
        unique_words_stemm.remove(word)

number_of_unique_words = len(unique_words_stemm)
print(number_of_unique_words)

p_word = {w:0 for w in unique_words_stemm}

for word in unique_words_stemm:
    p_word[word] = count_of_words[word] 
    
p_header = {h: 0 for h in header_list}

for head in header_list:
    p_header[head] = headers[head] / number_of_articles
    
print("начало создание pandas")

import pandas as pd
import numpy as np
data = np.zeros((10, len(unique_words_stemm)))
p_word_by_head = pd.DataFrame(data, index = header_list, columns = unique_words_stemm)

for article in articles:
    words_article = article.split()
    head = words_article.pop(0)
    
    for w in words_article:
        w = stemmer.stem(w)
        p_word_by_head.loc[head, w] = p_word_by_head.loc[head, w] + 1
        

print("вероятности посчитаны")
print("начало обработки тестовых")

g = open('S:/Download/ML/news_test.txt', encoding='utf-8')
test_atr = g.readlines()

all_test_articles = []

for a in articles:
    a = a.lower()
    all_test_articles.append(a)
    
test = []

for a in all_articles:
    for symbol in symbols: 
        a = a.replace(str(symbol),'')
    test.append(a)
    
print("тестовые тексты обработаны")
    
test_stemm = []

p_head_by_word = {h:0 for h in header_list}

#articles[0] = articles[0][1:]

test_answer_headers = []

for article in test:
    p_head_by_word = {h:0 for h in header_list}
    
    words_article = article.split()
    
    for head in header_list:
        for w in words_article:
            w = stemmer.stem(w)
            if (w in unique_words_stemm):
                probability = p_word_by_head.loc[head, w] * p_header[head] / p_word[w]
                p_head_by_word[head] = p_head_by_word[head] + probability
        print(head)

    this_header = max(p_head_by_word.keys(), key=(lambda k: p_head_by_word[k]))
    test_answer_headers.append(this_header)
    print("статья обработана")
    
    
with open("S:/Download/ML/output.txt", "w") as file:
    print(test_answer_headers, file=file, sep="\n")