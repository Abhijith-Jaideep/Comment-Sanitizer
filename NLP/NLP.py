from flask import Flask, request, jsonify
import json
import re
import string
import pickle

# pandas is deliberately not imported. It cost 2.06s of a measured 11.06s
# cold start and was used only to wrap a single comment in a one-row
# DataFrame before calling .apply(). The vectorizer takes any iterable of
# strings, so a plain list does the same job for free. On a 512MB free tier
# the memory it was holding matters too.

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


def make_test_predictions(comment_text):
    # Same pipeline as before, minus the DataFrame round trip. The decision
    # rule is unchanged on purpose: the six label probabilities are summed and
    # compared against 1, so predictions match the previous build exactly.
    X_test_transformed = td.transform([clean_text(comment_text)])
    y_test_pred = mp.predict_proba(X_test_transformed)
    result = sum(y_test_pred[0])
    if result >= 1:
        return 1
    else:
        return 0

@app.route("/health", methods=['GET'])
@app.route("/", methods=['GET'])
def health():
    # Exists so the container can be woken before anyone needs a verdict.
    # Flask only starts serving once the pickles at the top of this file are
    # loaded, so a 200 here means the model is ready, not just the process.
    return jsonify({"status": "ok"})


@app.route("/", methods=['POST'])
def sanitize():
    val = request.get_json()
    val = json.loads(val['body'])
    val = val['comment']

    comment_text = val
    result = make_test_predictions(comment_text)
    if(result==0):
        return(jsonify({"msg": 'Positive'}))
    else:
        return(jsonify({"msg": 'Negative'}))

    

if('__main__' == __name__):
    app.run(debug=True, port=7000,use_reloader=False,use_debugger=False)
