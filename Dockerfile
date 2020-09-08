FROM ubuntu:18.04

WORKDIR /root

RUN apt-get update && apt-get install -y python3 git python3-pip nano
RUN pip3 install torch transformers numpy flask
RUN apt-get install curl -y
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get update && apt-get install git-lfs -y
RUN apt-get install vim -y 
RUN git clone https://github.com/ls-da3m0ns/Bert-sentiment-analysis.git

WORKDIR ./Bert-sentiment-analysis

CMD ["python3","./app.py"]
