#!/usr/bin/env python
# coding: utf-8

# Loading various packages needed for the model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from string import punctuation
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


# Scrapping news data from Politifact news website

import urllib.request,sys,time
from bs4 import BeautifulSoup
import requests


# =============================================================================
# pagesToGet= 400
# 
# upperframe=[]  
# for page in range(1,pagesToGet+1):
#     print('processing page :', page)
#     url = 'https://www.politifact.com/factchecks/list/?page='+str(page)
#     print(url)
#     
#     #an exception might be thrown, so the code should be in a try-except block
#     try:
#         #use the browser to get the url. This is suspicious command that might blow up.
#         page=requests.get(url)                             # this might throw an exception if something goes wrong.
#     
#     except Exception as e:                                   # this describes what to do if an exception is thrown
#         error_type, error_obj, error_info = sys.exc_info()      # get the exception information
#         print ('ERROR FOR LINK:',url)                          #print the link that cause the problem
#         print (error_type, 'Line:', error_info.tb_lineno)     #print error info and line that threw the exception
#         continue                                              #ignore this page. Abandon this and go back.
#     time.sleep(2)   
#     soup=BeautifulSoup(page.text,'html.parser')
#     frame=[]
#     links=soup.find_all('li',attrs={'class':'o-listicle__item'})
#     print(len(links))
#     filename="NEWS.csv"
#     f=open(filename,"w", encoding = 'utf-8')
#     headers="Statement,Link,Date, Source, Label\n"
#     f.write(headers)
#     
#     for j in links:
#         Statement = j.find("div",attrs={'class':'m-statement__quote'}).text.strip()
#         Link = "https://www.politifact.com"
#         Link += j.find("div",attrs={'class':'m-statement__quote'}).find('a')['href'].strip()
#         Date = j.find('div',attrs={'class':'m-statement__body'}).find('footer').text[-14:-1].strip()
#         Source = j.find('div', attrs={'class':'m-statement__meta'}).find('a').text.strip()
#         Label = j.find('div', attrs ={'class':'m-statement__content'}).find('img',attrs={'class':'c-image__original'}).get('alt').strip()
#         frame.append((Statement,Link,Date,Source,Label))
#         f.write(Statement.replace(",","^")+","+Link+","+Date.replace(",","^")+","+Source.replace(",","^")+","+Label.replace(",","^")+"\n")
#     upperframe.extend(frame)
# f.close()
# data=pd.DataFrame(upperframe, columns=['Statement','Link','Date','Source','Label'])
# data.head()
# =============================================================================



# Saving the scrapped data as a CSV file.

#data.to_csv('News_data.csv',index=False)

# Reading the news data

df = pd.read_csv('news_data.csv')
df.head()


df = df.drop('Link',1) # Dropping link column as it is not be of use in the analysis


# Checking for data type, null values in the dataframe 

df.info()



# Distribution of various news categories in the data

df.Label.value_counts()



# Dropping half-flip, no-flip and full-flop news categories from the data.

df = df[(df['Label']!='no-flip') & (df['Label']!='half-flip') & (df['Label']!='full-flop')]


label_dist = df.Label.value_counts()
label_dist


# Plot to visualize count of various news categories

plt.figure(figsize=[11,9])
sns.set(style='darkgrid')
sns.countplot(x='Label', data=df,order = df['Label'].value_counts().index)
plt.show()


# Splitting year from data column to visualize distribution of various categories of data over the year

df['Date'] = df['Date'].apply(lambda x: x.split(','))
df['Year'] = df['Date'].apply(lambda x: x[1])


false_df = df[df['Label']=='false']
false_df['Year'].value_counts().plot.bar(color='red', figsize=[8,6])
plt.xlabel('Year')
plt.ylabel('count of news')
plt.title('Fake news over the year')
plt.show()


barelytrue_df = df[df['Label']=='barely-true']
barelytrue_df['Year'].value_counts().plot.bar(color='orange', figsize=[8,6])
plt.xlabel('Year')
plt.ylabel('count of news')
plt.title('Barely true news over the year')
plt.show()


true_df = df[df['Label']=='true']
true_df['Year'].value_counts().plot.bar(figsize=[8,6])
plt.xlabel('Year')
plt.ylabel('count of news')
plt.title('True news over the year')
plt.show()


#Creating a dataframe after grouping the data on Label and Source

groupby_label = pd.DataFrame(df.groupby(['Label','Source'],as_index=False).count())
groupby_label


# Plotting graphs for various news sources

label = df.Label.unique()

for i in label:

    news_source = groupby_label[groupby_label['Label']==i].nlargest(10,'Statement')

    plot = sns.catplot(y='Statement', x='Source', data = news_source, kind='bar', aspect=2, height=8)    
    plot.set_xticklabels(rotation=40)
    plot.fig.suptitle(str(i)+' newsource')
    plt.show()


# ### Cleaning text for model building 


stop = set(stopwords.words('english'))
stop.update(punctuation)
stop.update(("’","'s","”","“","‘","–",'``',"Says","''"))
print(stop)


# Function to clean text

lemmatizer = WordNetLemmatizer()
def clean_review(text):
    clean_text = []
    for w in word_tokenize(text):
        if w.lower() not in stop:
            new_w = lemmatizer.lemmatize(w)
            clean_text.append(new_w)
    return clean_text

def join_text(text):
    return " ".join(text)



df.Statement = df.Statement.apply(clean_review)
df.Statement = df.Statement.apply(join_text)


# ### Visualizing various categories of news and the frequent words present


import collections

df['temp'] = df['Statement'].apply(lambda x:str(x).split())
top = collections.Counter([item for sublist in df['temp'] for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words','count']
temp = temp[(temp.Common_words!= "Says")]
temp.style.background_gradient(cmap='Blues')


import plotly.express as px

fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Statement', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()


true = df[df['Label']== 'true']

# MosT common words in True labelled news
top = collections.Counter([item for sublist in true['temp'] for item in sublist])
temp_true = pd.DataFrame(top.most_common(21))
temp_true.columns = ['Common_words','count']
temp_true = temp_true[(temp_true.Common_words!= "Says")]
temp_true.style.background_gradient(cmap='Greens')



fig = px.bar(temp_true, x="count", y="Common_words", title='Most Commmon Words present in True news', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()


fake = df[df['Label']== 'false']

#MosT common positive words
top = collections.Counter([item for sublist in fake['temp'] for item in sublist])
temp_fake = pd.DataFrame(top.most_common(21))
temp_fake.columns = ['Common_words','count']
temp_fake = temp_fake[(temp_fake.Common_words!= "Says")]
temp_fake.style.background_gradient(cmap='Reds')



fig = px.bar(temp_fake, x="count", y="Common_words", title='Most Commmon Words present in False news', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()


# ### Visalizing word cloud for true and false news 

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

stop_words = ['Says']+list(STOPWORDS)

plt.figure(figsize = (20,20)) 
word_cld = WordCloud(max_words = 1000 ,contour_color= 'Yellow', width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(fake.Statement))
plt.imshow(word_cld , interpolation = 'bilinear')
plt.show()


stop_wordss = ['Says'+'percent'+"Say"+'State']+list(STOPWORDS)

plt.figure(figsize = (20,20))
word_cld_1 = WordCloud(max_words = 1000 ,contour_color= 'Yellow', width = 1600 , height = 800 , stopwords = stop_wordss).generate(" ".join(true.Statement))
plt.imshow(word_cld_1 , interpolation = 'bilinear')
plt.show()


# Function to produce ngrams words visuals

def get_top_text_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# Bigrams for fake news

plt.figure(figsize = (16,9))
most_common_bi = get_top_text_ngrams(fake.Statement,10,2)
most_common_bi = dict(most_common_bi)
sns.barplot(x=list(most_common_bi.values()),y=list(most_common_bi.keys()))


# Bigrams for True news

plt.figure(figsize = (16,9))
most_common_bi = get_top_text_ngrams(true.Statement,10,2)
most_common_bi = dict(most_common_bi)
sns.barplot(x=list(most_common_bi.values()),y=list(most_common_bi.keys()))



fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
word=df[df['Label']=='true']['Statement'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='green')
ax1.set_title('True News text')
word_1=df[df['Label']=='false']['Statement'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word_1.map(lambda x: np.mean(x)),ax=ax2,color='red')
ax2.set_title('Fake news text')
fig.suptitle('Average word length in each text')


# replacing string values in Label column with integr
df['Label'] = df['Label'].replace({'false':0,'barely-true':1, 'mostly-true':2,'half-true':3,'pants-fire':4,'true':5})

# ### Model Building 


X = df['Statement']
y = df['Label']

# Splitting the data in 80:20 split for train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)


count_vector = CountVectorizer(max_features=40)
pickle.dump(count_vector, open('count_vec.pkl','wb'))
count_vec = pickle.load(open('count_vec.pkl','rb'))


x_train_features = count_vec.fit_transform(X_train).todense()
x_test_features = count_vec.transform(X_test).todense()


from sklearn.multiclass import OneVsRestClassifier

logstcre = LogisticRegression()
ovr = OneVsRestClassifier(logstcre)

ovr.fit(x_train_features, y_train)
predicted_lr = ovr.predict(x_test_features)
print("LogisticRegression classification chart:\n",classification_report(y_test, predicted_lr))



from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(random_state=20)

rf_clf.fit(x_train_features, y_train)
predicted_rf = rf_clf.predict(x_test_features)
predicted_rf
print("Random forest classification chart:\n",classification_report(y_test, predicted_rf))



nb_clf = MultinomialNB()
nb_clf.fit(x_train_features, y_train)

pickle.dump(nb_clf, open('model.pkl','wb'))   # Saving the model

model = pickle.load(open('model.pkl','rb'))
y_pred = model.predict(x_test_features)
print(classification_report(y_test, y_pred))

