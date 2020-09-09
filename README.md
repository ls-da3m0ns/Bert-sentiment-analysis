# Bert Sentiment Analysis #
### Introduction ###

### how to use ###
* make sure you have docker installed
* then use next command to build the docker image <b> docker build -t bert-docker . </b>
* to run this container use <b> docker run -d -p 9999:9999 -e PORT=9999 bert-docker:latest </b>

### Requirements ### 
* Pytorch
* Flask
* Python
* HTML/CSS/JS

### Demo Images ###
<div align="center">
  <p><b>Index page which take sentence as a input. </b></p>
  <img src="./images/bert-demo.PNG" width="600">  
</div>

<div align="center">
  <br>
  <p> <b> Output page to display the prediction </b></p>
  <img src="./images/bert-demo-ouput.PNG" width="600">  
</div>

### Accuracy ###

<div align="center">
  <img src="./images/accuracy.png" width="600">  
</div>

