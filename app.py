from src import config 
import torch 
import time
import torch.nn as nn

import flask 
from flask import Flask,request
from src.model import BERTBaseUncased
from flask import render_template

app = Flask(__name__)

MODEL = None
DEVICE = "cpu"


def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    
    sentence = str(sentence)
    sentence = " ".join(sentence.split())
    
    inputs = tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length = max_len,
    )
    
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    
    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)
    
    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)
    
    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]

    

@app.route("/predict")
def predict(sentence):
    #sentence = request.args.get("sentence")
    start_time = time.time()
    positive_prediction = sentence_prediction(sentence)
    negative_prediction = 1 - positive_prediction
    response = {}
    response["response"] = {
        'positive': str(positive_prediction),
        'negative': str(negative_prediction),
        'time_taken': str(time.time() - start_time),
    }
    
    return render_template("outputs.html",positive=positive_prediction,negative=negative_prediction,time_taken=str(time.time() - start_time))
   # return flask.jsonify(response)


@app.route("/",methods=["GET"])
def start_page():
    return render_template("index.html")

@app.route("/",methods=["POST"])
def input_accept():
    text = request.form['text']
    print(text.upper())
    return predict(text)

if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(host="127.0.0.1", port="9999",debug=True)