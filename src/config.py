import transformers

MAX_LEN = 512 
TRAIN_BATCH_SIZE = 14
VALID_BATCH_SIZE = 8
EPOCHS = 3
BASE_PATH = "/root/Bert-sentiment-analysis/"
BERT_PATH = "bert-base-uncased"
MODEL_PATH = BASE_PATH + "src/model.bin"
TRAINING_FILE = BASE_PATH + "/input/imdb.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH,do_lower_case = True)
UPLOAD_FOLDER = './input/'