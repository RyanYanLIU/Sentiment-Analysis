import os 
import pandas as pd 
from textblob import TextBlob

link = 'C:\\Python Software\\Sentiment Analysis\\movie\\train\\pos'
path = os.listdir(link)
list1 = []
positive = 0
neural = 0
negative = 0
for i in path:
    position = link + '\\' + i
    with open(position, "r",encoding='utf-8') as f:
        data = f.read()   #Read Txt
        #df = df.append([data], ignore_index = False)
        list1.append(data)
    text = TextBlob(data)
    sentiment = text.sentiment.polarity
    if sentiment == 0:
        neural += 1
    elif sentiment > 0:
        positive += 1
    elif sentiment < 0:
        negative += 1
print('Positive {} \n Neural {} \n Negative {}'.format(positive, neural, negative))