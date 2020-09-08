import transformers

MAX_LEN = 512 
TRAIN_BATCH_SIZE = 14
VALID_BATCH_SIZE = 8
EPOCHS = 3
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../input/imdb.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH,do_lower_case = True)