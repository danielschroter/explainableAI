import newspaper
import requests
from newspaper import Article
import nltk.data
from toolbox.data_preparation import get_data, sent_tokenize_text

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

import requests
import json

def get_text_for_keyword(key_word):

    url = f"https://gnews.io/api/v4/search?token=48090e4a4d7fc106cbf654410c11d535&q={key_word}&lang=en"

    payload={}
    headers = {
      'Cookie': 'PHPSESSID=7touf391ktjagssgojerpququd'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    print(response.text)

    articles = json.loads(response.text)['articles']

    article_urls = []
    print(len(articles))
    print(key_word)
    text = [];
    for a in articles:
        article_urls.append(a['url'])
        article=Article(a['url'])
        try:
            article.download()
            article.parse()
            text.append(article.text)
        except:
            continue

    text_total = ""
    for t in text:
        text_total = text_total + t


    return text_total

# Key words to search for, should represent the topics
key_words = ["suicide", "libertarianism", "human cloning", "private military", "child actors", "Guantanamo", "mandatory retirement", "nuclear weapon", "urbanization", "compulsory voting", "Homeschooling", "legalize cannabis", "prostitution", "flag burning", "women in combat", "journalism", "space exploration", "vow of celibacy", "marriage", "capital punishment", "intellectual property rights", "atheism"]

total_text = ""
for kw in key_words:
    text = get_text_for_keyword(kw);
    text = text.replace(".", ". ")
    sentences = sent_tokenize_text(text);
    full_text = "\n".join(sentences)
    total_text = total_text +"\n \n"+full_text



print(total_text)


with open("full_text_paragraph.txt", "w", encoding="utf-8") as text_file:
    text_file.write(total_text)








