import pandas as pd
import numpy as np
from datetime import date, timedelta
from collections import Counter
from operator import itemgetter
import os
import warnings

#Visualisation Library
import matplotlib.pyplot as plt
import cufflinks as cf
import seaborn as sns
from wordcloud import WordCloud 

#Preprocessing Libraries
from nltk.corpus import stopwords
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Topic Modelling Libraries

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer, TfidfTransformer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()
dataset = pd.read_csv('../input/reddit-vaccine-myths/reddit_vm.csv', error_bad_lines=False);
dataset.shape
dataset.head()
#Creation of StopWords

stop_words = stopwords.words('english') #Call the StopWords Function from the Library
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'op']) #I am removing some extra words like 'OP' and 'Edu' by extending the stopwords list
stop_words[0:5]
def pre_process(s):
    s = s.str.lower()
    s = s.str.replace(r'(?i)\brt\b', "")
    s = s.str.replace(' via ',"") 
    s = s.replace(r'@\w+', "", regex = True)
    s = s.replace(r'http\S+', '', regex = True)
    s = s.replace(r'www.[^ ]+', '', regex = True)
    s = s.replace(r'[0-9]+', '', regex = True)
    s = s.replace(r'''[¬!"#$%&()*+,-./:;<=>?@[\]’^'_`\{|}~]''', '', regex=True)
    return s

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def lemmatizing(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def final_text(words):
     return ' '.join(words)
     dataset.body = pre_process(dataset['body']) 
dataset = dataset.dropna(subset = ['body']) #Drop Empty Rows

dataset['token'] = remove_stopwords(dataset['body']) 
dataset['token'] = dataset['token'].apply(lambda x: lemmatizing(x)) 
dataset['body_combined_text'] = dataset['token'].apply(lambda x: final_text(x))

dataset[dataset['body_combined_text'].duplicated(keep=False)].sort_values('body_combined_text').head() #View Duplicates
dataset = dataset.drop_duplicates(['body_combined_text']) #Remove Duplicates

#Plot Comments Per Month

dataset['date'] = pd.to_datetime(dataset['timestamp'])
dataset['Hour'] = dataset['date'].apply(lambda x: x.hour)
dataset['Month'] = dataset['date'].apply(lambda x: x.month)
dataset['Month'] = dataset['Month'].replace({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun', 7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
dataset_month = dataset.Month.value_counts().reindex(['Jan', 'Feb', 'Mar', 'Apr', "May", 'Jun','Jul', "Aug", "Sep", "Oct", "Nov", "Dec"])
dataset_month.iplot()
a = dataset['token']
a = [x for i in a for x in i]
top_20 = pd.DataFrame(Counter(a).most_common(20), columns=['word', 'freq']) #Check Word Frequency via Dataframe sorted by mos frequent.
print(top_20)
no_of_unique_words = len(set(a)) #Check Number of Unique Words in the Dataset.
print("There are " + str(no_of_unique_words) + " unique words in the dataset.")
#Functions For Analysit

def top_words(topic, n_top_words):
    return topic.argsort()[:-n_top_words - 1:-1]  


def topic_table(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        t = (topic_idx)
        topics[t] = [feature_names[i] for i in top_words(topic, n_top_words)]
    return pd.DataFrame(topics)

coherence_scores = []

def find_cv(): #Find Number of K
    for num in topic_nums:
        nmf = Nmf(corpus=corpus, num_topics=num, id2word=dictionary,normalize=True)
        cm = CoherenceModel(model=nmf, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_scores.append(round(cm.get_coherence(), 5))

def compare_cv():
    for m, cv in zip(topic_nums, coherence_scores):
        print("K =", m, " CV: ", round(cv, 2))
    scores = list(zip(topic_nums, coherence_scores))
    best_cv = sorted(scores, key=itemgetter(1), reverse=True)[0][0]
    print('\n')
    return best_cv
topic_nums = list(np.arange(5, 55 + 1, 3)) #The range that we will run NMF on
texts = dataset.token
dictionary = Dictionary(texts) #create dictionary 
dictionary.filter_extremes(no_below=5, no_above=0.8, keep_n=2000) 
corpus = [dictionary.doc2bow(text) for text in texts] 
#Call the Functions
find_cv()  
#Store the return value of compare_cv function to best_cv to use in the NMF
best_cv = compare_cv()
tfidf_vectorizer = TfidfVectorizer(min_df=3, max_df=0.85, max_features=5000, ngram_range=(1, 2), preprocessor=' '.join)
tfidf = tfidf_vectorizer.fit_transform(texts)
tfidf_fn = tfidf_vectorizer.get_feature_names()
nmf = NMF(n_components= best_cv , init='nndsvd', solver='cd',random_state=42).fit(tfidf)
docweights = nmf.transform(tfidf_vectorizer.transform(texts))

n_top_words = 10

topic_df = topic_table(nmf, tfidf_fn, n_top_words).T
topic_df

topic_df['topics'] = topic_df.apply(lambda x: [' '.join(x)], axis=1) 
topic_df['topics'] = topic_df['topics'].str[0] 

topic_df = topic_df['topics'].reset_index()
topic_df.columns = ['topic_num', 'topics']

topic_df.head()
docweights = nmf.transform(tfidf_vectorizer.transform(texts))

n_top_words = 10

topic_df = topic_table(nmf, tfidf_fn, n_top_words).T #Creates a Topic Table to see the Top Words in each topic number.
topic_df
topic_df['topics'] = topic_df.apply(lambda x: [' '.join(x)], axis=1) #Joins all words in one column for easy interpretation of topics.
topic_df['topics'] = topic_df['topics'].str[0]  

topic_df = topic_df['topics'].reset_index()
topic_df.columns = ['topic_num', 'topics']

topic_df.head()
dataset['topic_num'] = docweights.argmax(axis=1)
dataset = dataset.merge(topic_df[['topic_num','topics']],"left") #Merge the Topic dataset to our main dataset to find which Reddit post belongs to which topic
columns = ['score', 'url', 'comms_num', 'created', 'body_combined_text', 'date', 'Hour', 'Month']
dataset.drop(columns = columns, inplace = True) #Removes Unncessary Columns
dataset.head()
A = tfidf_vectorizer.transform(texts)
W = nmf.components_
H = nmf.transform(A)

print('A = {} x {}'.format(A.shape[0], A.shape[1]))
print('W = {} x {}'.format(W.shape[0], W.shape[1]))
print('H = {} x {}'.format(H.shape[0], H.shape[1]))
r = np.zeros(A.shape[0])

for row in range(A.shape[0]):
    r[row] = np.linalg.norm(A[row, :] - H[row, :].dot(W), 'fro')

sum_sqrt_res = round(sum(np.sqrt(r)), 3)
print(sum_sqrt_res)

dataset['resid'] = r
resid_data = dataset[['topic_num','resid']].groupby('topic_num').mean().sort_values(by='resid') 


resid_data.iplot( kind = 'bar', title = 'Average Residuals by Topic', xTitle = 'Topic Number', yTitle = 'Residuals')
resid_data.head()
def word_cloud(df_weights, n_top_words=20, is_print=True, is_plot=True):
    s_word_freq = pd.Series(df_weights['count'])
    s_word_freq.index = df_weights['word']
    di_word_freq = s_word_freq.to_dict()
    cloud = WordCloud(width=1600, height=800, background_color='white').generate_from_frequencies(di_word_freq)
    plt.figure(1,figsize=(13, 10))
    if is_print:
        print(df_weights.iloc[:n_top_words,:])
    if is_plot:
        plt.imshow(cloud)
        plt.axis('off')
        plt.show()
    return cloud
dataset['joined'] = dataset['token'].apply(lambda x: final_text(x))
frequent_NN = pd.Series(' '.join(dataset['joined']).split()).value_counts()
cv = CountVectorizer(max_df = 0.6, min_df = 10, max_features=None, ngram_range=(1,4))
X = cv.fit_transform(dataset['joined'])
cvec = cv.fit(dataset.joined)
bag_of_words = cvec.transform(dataset.joined)
feature_names = cvec.get_feature_names()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(bag_of_words)
word_cnts = np.asarray(bag_of_words.sum(axis=0)).ravel().tolist()  # for each word in column, sum all row counts
df_cnts = pd.DataFrame({'word': feature_names, 'count': word_cnts})
df_cnts = df_cnts.sort_values('count', ascending=False)
weights = np.asarray(tfidf.mean(axis=0)).ravel().tolist()
df_weights = pd.DataFrame({'word': feature_names, 'weight': weights})
df_weights = df_weights.sort_values('weight', ascending=False)

df_weights = df_weights.merge(df_cnts, on='word', how='left')
df_weights = df_weights[['word', 'count', 'weight']]
topic_no_0 = dataset[dataset['topic_num'] == 0]
topic_no_0 = dataset[dataset['topic_num'] == 0] #Filter to get only those comments which belongs to Topic 0
frequent_NN = pd.Series(' '.join(topic_no_0['joined']).split()).value_counts()  #Creates a new dataframe that joins and count the frequency of the number
cv = CountVectorizer(max_df = 0.2, min_df = 2, max_features=None, ngram_range=(1,2))
X = cv.fit_transform(topic_no_0['joined'])
cvec = cv.fit(topic_no_0.joined)
bag_of_words = cvec.transform(topic_no_0.joined)
feature_names = cvec.get_feature_names()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(bag_of_words)

word_counts = np.asarray(bag_of_words.sum(axis=0)).ravel().tolist()
df_cnts = pd.DataFrame({'word': feature_names, 'count': word_counts})
df_cnts = df_cnts.sort_values('count', ascending=False)
weights = np.asarray(tfidf.mean(axis=0)).ravel().tolist()


df_weights = pd.DataFrame({'word': feature_names, 'weight': weights})
df_weights = df_weights.sort_values('weight', ascending=False)
df_weights
df_weights = df_weights.merge(df_cnts, on='word', how='left')
df_weights = df_weights[['word', 'count', 'weight']]
df_weights
cloud_all = word_cloud(df_weights, is_print=True)
