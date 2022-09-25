from flask import Flask, request, jsonify
import json
import pandas as pd
import re
import string
import pickle

app = Flask(__name__)

with open('model_pickle','rb') as file:
    mp = pickle.load(file)

with open('vectorizer.pk','rb') as file:
    td = pickle.load(file)


def  clean_text(text):

    text =  text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text = re.sub("(\\W)"," ",text) 
    text = re.sub('\S*\d\S*\s*','', text)
    return text


def make_test_predictions(df):
    df.comment_text = df.comment_text.apply(clean_text)
    X_test = df.comment_text
    X_test_transformed = td.transform(X_test)
    y_test_pred = mp.predict_proba(X_test_transformed)
    result =  sum(y_test_pred[0])
    if result >=1 :
       return 1
    else :
       return 0

@app.route("/", methods=['POST'])
def sanitize(): 
    val = request.get_json()
    val = json.loads(val['body'])
    val = val['comment']
    
    comment_text = val
    comment ={'comment_text':[comment_text]}
    comment = pd.DataFrame(comment)
    result = make_test_predictions(comment)
    print(result)
    if(result==0):
        return(jsonify({"msg": 'Positive'}))
    else:
        return(jsonify({"msg": 'Negative'}))

    

if('__main__' == __name__):
    app.run(debug=True, port=7000,use_reloader=False,use_debugger=False)
