from textblob import TextBlob
from newspaper import Article
import nltk

url = 'https://everythingcomputerscience.com'
article = Article(url = url)

article.download()
article.parse()
#nltk.download('punkt')
article.nlp()

txt = article.summary
#TextBlob
obj = TextBlob(txt)
sentiment = obj.sentiment.polarity

if sentiment == 0:
    print('Neural')
elif sentiment > 0:
    print('Positive')
else:
    print('Negative')