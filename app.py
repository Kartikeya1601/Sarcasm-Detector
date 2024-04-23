from flask import Flask, render_template, request
import re
import string
import nltk
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import google.generativeai as genai
from bs4 import BeautifulSoup
from markdown import markdown

app = Flask(__name__)
#model = pickle.load(open('model.pkl','rb'))
model = load_model('model.h5')
stopwords = nltk.corpus.stopwords.words('english')
tok = Tokenizer(num_words=1000)

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




# Prediction function for LSTM model
def predict_sarcasm_lstm(text):
    max_len = 150
    # Preprocess the text
    text = clean_text(text)
    # Tokenize and pad the sequence
    tok = Tokenizer(num_words=1000)
    token = tok.texts_to_sequences([text])
    padded = sequence.pad_sequences(token, maxlen=max_len)
    # Make prediction
    prediction = model.predict(padded)
    return prediction[0][0]

def markdown_to_text(markdown_text):
    html = markdown(markdown_text)
    soup = BeautifulSoup(html, features="html.parser")
    return soup.get_text(separator="\n")


# Prediction function for GenerativeModel
def predict_sarcasm_generative(query):
    
    GOOGLE_API_KEY = 'AIzaSyDOStnx6lj2AV4JXOnUR1viPhcMzWKA5mQ'
    genai.configure(api_key=GOOGLE_API_KEY)

    g_model = genai.GenerativeModel('gemini-pro')
    query = query+". Is this a sarcastic comment? Give one word answer."
    res = g_model.generate_content(query)
    x = markdown_to_text(res.text)+""
    if (x == "No"):
        return "Prediction: Not a sarcastic Message."
    else:
        return "Prediction: Sarcastic Message."


@app.route('/predict', methods=['POST'])
def predict():
   query = request.form['head'];
   lstm_prediction = predict_sarcasm_lstm(query)
   
   if lstm_prediction>=0.55:
       message = "Prediction: Sarcasm Found"
   else:
       message = "Prediction: No Sarcasm Detected."

   generative_prediction = predict_sarcasm_generative(query)

   return render_template("templates/output.html", x = message,y = generative_prediction,
                          lstm_model_name = "Recurrent Neural Network",
                          gemini_model_type="Gemini-1.5-pro-latest")

@app.route('/')
def hello_world():
    return render_template('/index.html')


if __name__ == '__main__':
    app.run(debug=True,port =5555)
    app.run()

