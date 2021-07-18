from src import config 
import torch 
import time
import os
import torch.nn as nn

import flask 
from flask import Flask,request
from src.model import BERTBaseUncased
from flask import render_template
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER

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

    return render_template("outputs.html",positive=positive_prediction,negative=negative_prediction,time_taken=str(time.time() - start_time))

@app.route('/fileupload', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if 'tmp_filename' not in request.files:
            return 'there is no tmp_filename in form!'

        file1 = request.files['tmp_filename']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)

        img = cv2.imread(path)
        """
        @TODO pass this img to predict function of your model
        """
        prediction = "something_tmp" #call your model here
        return prediction
    return '''
    <h1>Upload new File</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="tmp_filename">
      <input type="submit">
    </form>
    '''



@app.route('/textupload', methods=['GET','POST'])
def upload_text():
    if request.method == 'POST':
        text = request.form['text']
        processed_text = text.upper()
        """
        @TODO pass this text to predict function of your model
        """
        prediction = "something_tmp" #call your model here
        return prediction
    return '''
    <h1>Enter text below</h1>
   <form method="POST">
    <input name="text">
    <input type="submit">
    </form>
    '''


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
    print("PORT got from port variable {}".format(config.PORT))
    app.run(host="0.0.0.0", port=config.PORT,debug=True)
