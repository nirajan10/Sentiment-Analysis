from flask import Flask, request, url_for, redirect, render_template
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re, pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

model = pickle.load(open('./model/final_LR.pkl', 'rb'))
tfidf = pickle.load(open('./model/tfidf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("sentiment_analysis.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    text = request.form['review']
    corpus = []
    soup = BeautifulSoup(text, "html.parser")
    review = soup.get_text()
    review = re.sub('\[[^]]*\]', ' ', review)
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    lem = WordNetLemmatizer()
    review = [lem.lemmatize(word) for word in review]
    review = ' '.join(review)
    corpus.append(review)
    review_tf_idf = tfidf.transform(corpus)
    review_tf_idf.toarray()
    
    output = model.predict(review_tf_idf)

    if output.item() == 1:
        return render_template('sentiment_analysis.html',pred='Positive', review=review)
    else:
        return render_template('sentiment_analysis.html', pred='Negative', review=review)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int("3000"), debug=True)