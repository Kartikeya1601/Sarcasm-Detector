import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import re
import nltk
import itertools
from collections import Counter
import nltk.stem
from nltk.stem import WordNetLemmatizer
from wordcloud import STOPWORDS
import string
import numpy as np
from bs4 import BeautifulSoup
from markdown import markdown
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

lemm = WordNetLemmatizer()
pd.set_option('display.max_colwidth', -1)
sar_acc = pd.read_json("D:/Sem-8 Submissions/NLP/Sarcasm_Headlines_Dataset.json",lines=True)
sar_acc['source'] = sar_acc['article_link'].apply(lambda x: re.findall(r'\w+', x)[2])

pd.set_option('display.max_colwidth', -1)
sar_acc_1 = pd.read_json("D:/Sem-8 Submissions/NLP/Sarcasm_Headlines_Dataset_v2.json",lines=True)
sar_acc_1['source'] = sar_acc['article_link'].apply(lambda x: re.findall(r'\w+', x)[2])


sar_acc_1 = sar_acc_1[sar_acc.columns]
sar_acc = pd.concat([sar_acc, sar_acc_1], ignore_index=True)

all_words = sar_acc['headline'].str.split(expand=True).unstack().value_counts()
words = sar_acc['headline'].str.split(expand=True).unstack()

sar_det = sar_acc[sar_acc.is_sarcastic==1]
sar_det.reset_index(drop=True, inplace=True)
acc_det = sar_acc[sar_acc.is_sarcastic==0]
acc_det.reset_index(drop=True, inplace=True)

# Tokenizing the Headlines of Sarcasm
sar_news = []
for rows in range(0, sar_det.shape[0]):
    head_txt = sar_det.headline[rows]
    head_txt = head_txt.split(" ")
    sar_news.append(head_txt)

#Converting into single list for Sarcasm
sar_list = list(itertools.chain(*sar_news))

# Tokenizing the Headlines of Acclaim
acc_news = []
for rows in range(0, acc_det.shape[0]):
    head_txt = acc_det.headline[rows]
    head_txt = head_txt.split(" ")
    acc_news.append(head_txt)
    
#Converting into single list for Acclaim
acc_list = list(itertools.chain(*acc_news))



stopwords = nltk.corpus.stopwords.words('english')
sar_list_restp = [word for word in sar_list if word.lower() not in stopwords]
acc_list_restp = [word for word in acc_list if word.lower() not in stopwords]


#Data cleaning for getting top 30
sar_cnt = Counter(sar_list_restp)
acc_cnt = Counter(acc_list_restp)

#Dictonary to Dataframe
sar_cnt_df = pd.DataFrame(list(sar_cnt.items()), columns = ['Words', 'Freq'])
sar_cnt_df = sar_cnt_df.sort_values(by=['Freq'], ascending=False)
acc_cnt_df = pd.DataFrame(list(acc_cnt.items()), columns = ['Words', 'Freq'])
acc_cnt_df = acc_cnt_df.sort_values(by=['Freq'], ascending=False)

#Top 30
sar_cnt_df_30 = sar_cnt_df.head(30)
acc_cnt_df_30 = acc_cnt_df.head(30)



sar_wost_lem = []
for batch in sar_news:
    sar_list_restp = [word for word in batch if word.lower() not in stopwords]
    lemm = WordNetLemmatizer()
    sar_list_lemm =  [lemm.lemmatize(word) for word in sar_list_restp]
    sar_wost_lem.append(sar_list_lemm)

#Acclaim headline after Lemmatization
acc_wost_lem = []
for batch in acc_news:
    acc_list_restp = [word for word in batch if word.lower() not in stopwords]
    lemm = WordNetLemmatizer()
    acc_list_lemm =  [lemm.lemmatize(word) for word in acc_list_restp]
    acc_wost_lem.append(sar_list_lemm)
    
from sklearn.feature_extraction.text import CountVectorizer
vec = []
for block in sar_wost_lem:
    vectorizer = CountVectorizer(min_df=0)
    sentence_transform = vectorizer.fit_transform(block)
    vec.append(sentence_transform)
    


## Number of words in the text ##
sar_acc["num_words"] = sar_acc["headline"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
sar_acc["num_unique_words"] = sar_acc["headline"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
sar_acc["num_chars"] = sar_acc["headline"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
sar_acc["num_stopwords"] = sar_acc["headline"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

## Number of punctuations in the text ##
sar_acc["num_punctuations"] =sar_acc['headline'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##
sar_acc["num_words_upper"] = sar_acc["headline"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
sar_acc["num_words_title"] = sar_acc["headline"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the text ##
sar_acc["mean_word_len"] = sar_acc["headline"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


#Getting X and Y ready
X = sar_acc.headline
Y = sar_acc.is_sarcastic
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.17,random_state=7)

max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(128)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Dense(128,name="FC2")(layer)
    layer = Activation('tanh')(layer)
    layer = Dropout(0.15)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(learning_rate=0.001),metrics=['accuracy'])

history = model.fit(sequences_matrix,Y_train,batch_size=95,epochs=15,validation_split=0.1)
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
model.save('model.h5')


def clean_text(sentences): 
    # convert text to lowercase 
    text = sentences.lower() 
    # remove text in square brackets 
    text = re.sub('\[.*?\]', '', text) 
    # removing punctuations 
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) 
    # removing words containing digits 
    text = re.sub('\w*\d\w*', '', text) 
    # Join the words 
    text = ' '.join([word for word in text.split() 
                     if word not in stopwords]) 
    return text 

#query = input("Enter a Sentence  ")
query = "mom starting to fear son's web series closest thing she will have to grandchild"
query = clean_text(query)
token = tok.texts_to_sequences([query])
padded = sequence.pad_sequences(token,maxlen=max_len)

mod = model.predict(padded)
print("\n",mod[0][0])
if(mod > 0.55):
    print(f"\n\nHeadline: {query}\nPrediction from RNN: A Saracastic Comment")
else:
    print(f"\n\nHeadline: {query}\nPrediction from RNN: Not a Saracastic Comment")


#########################################################################################
########GEMINI API##################################




GOOGLE_API_KEY = 'AIzaSyDOStnx6lj2AV4JXOnUR1viPhcMzWKA5mQ'
genai.configure(api_key=GOOGLE_API_KEY)

g_model = genai.GenerativeModel('gemini-1.5-pro-latest')


def markdown_to_text(markdown_text):
  html = markdown(markdown_text)
  soup = BeautifulSoup(html, features="html.parser")
  return soup.get_text(separator="\n")

query = query+". Is this a sarcastic comment? Give one word answer."
res = g_model.generate_content(query)
x = markdown_to_text(res.text)+""
if (x == "No"):
    print("Prediction from Gemini: Not a sarcastic Message.")
else:
    print("Prediction from Gemini: Sarcastic Message")
