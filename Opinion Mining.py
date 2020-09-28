import pandas as pd 
import re
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
warnings.filterwarnings('ignore')

data = pd.read_csv('Train_movie.csv',index_col = 0)
stopwords = set(stopwords.words('english'))

def remove_punctuation(text): 
    text = re.sub('[^a-zA-Z]',' ',text)
    words = text.lower().split()
    return ' '.join(words)

X = data['Message'].apply(remove_punctuation)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
Tfidf = TfidfVectorizer(ngram_range = (1,1),
                        stop_words = stopwords, max_df = 0.1, 
                        max_features = 200000, norm = 'l2')
lda = LatentDirichletAllocation(n_components = 10,
                                random_state = 120,
                                learning_method = 'online')

X_tfidf = Tfidf.fit_transform(X.values)
X_lda = lda.fit_transform(X_tfidf)

n_top_words = 4
feature_names = Tfidf.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i]
    for i in topic.argsort()[:-n_top_words:-1]]))
