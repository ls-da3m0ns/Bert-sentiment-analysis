import time
import os
from textblob import TextBlob

import flask 
from flask import Flask,request
from flask import render_template

from tempfile import NamedTemporaryFile
import cv2
import easyocr


app = Flask(__name__)

reader = None

def populate_reader():
    global reader
    reader = easyocr.Reader(['en'])

def paragraph_prediction(paragraph):
    data = dict()
    blob = TextBlob(paragraph)

    overall_polarity = blob.sentiment.polarity
    overall_subjectivity = blob.sentiment.subjectivity

    sentences = []
    for sentence in blob.sentences:
        sentences.append([str(sentence),
                f"{sentence.sentiment.polarity:.4f}",
                f"{sentence.sentiment.subjectivity:.4f}",
                sentence.sentiment_assessments.assessments])

    for i in range(len(sentences)):
        for j in range(len(sentences[i][-1])):
            ahem = sentences[i][-1][j]
            sentences[i][-1][j] = (ahem[0], f"{ahem[1]:.4f}", f"{ahem[2]:.4f}")

    return {
        "polarity": f"{overall_polarity:.4f}",
        "subjectivity": f"{overall_subjectivity:.4f}",
        "sentences": sentences
    }



@app.route('/fileupload', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if 'tmp_filename' not in request.files:
            return 'there is no tmp_filename in form!'

        file1 = request.files['tmp_filename']
        extention = os.path.splitext(file1.filename)
        with NamedTemporaryFile() as temp:
            iname = "".join([str(temp.name), extention[-1]])
            file1.save(iname)
            result = reader.readtext(iname)

        text_chunks = []
        for res in result:
            text_chunks.append(res[1])

        text = " ".join(text_chunks)
        prediction = paragraph_prediction(text)
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

        prediction = paragraph_prediction(text) #call your model here
        return prediction

    return '''
    <h1>Enter text below</h1>
   <form method="POST">
    <input name="text">
    <input type="submit">
    </form>
    '''


@app.route("/predict")
def predict(sentence):
    #sentence = request.args.get("sentence")
    start_time = time.time()
    paragraph_prediction(sentence)

    return render_template("outputs.html",positive=1,negative=2,time_taken=str(time.time() - start_time))


@app.route("/",methods=["GET"])
def start_page():
    return render_template("index.html")

@app.route("/",methods=["POST"])
def input_accept():
    text = request.form['text']
    print(text.upper())
    return predict(text)

if __name__ == "__main__":
    populate_reader()
    app.run(debug=True)
