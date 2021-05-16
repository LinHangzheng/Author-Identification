# -*- coding: utf-8 -*-
import os
import string
import torch
import csv
from transformers import BertTokenizer
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer 
from nltk.tokenize import word_tokenize 
import pickle
import re

# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

path = os.getcwd()
os.chdir(path)
os.listdir(path)
print (path) 

def remove_numbers(text): 
    result = re.sub(r'\d+', '', text) 
    return result 

def remove_punctuation(text): 
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator) 

def remove_whitespace(text): 
    return  " ".join(text.split()) 

# remove stopwords function 
def remove_stopwords(text): 
    stop_words = set(stopwords.words('english'))
    filtered_text = text
    for s in stop_words:
        filtered_text = re.sub(r'\b'+s+r'\b', '',filtered_text)
    return filtered_text 
  
# stem words in the list of tokenised words 
def stem_words(text): 
    stemmer = PorterStemmer() 
    word_tokens = word_tokenize(text) 
    stems = [stemmer.stem(word) for word in word_tokens] 
    return stems 
    
def pad(x, max_len):
    return x[:max_len] if len(x) > max_len else x + [0] * (max_len - len(x))

def encode_fn(text,length):
    sentence = tokenizer.encode(
                                text,                      
                                add_special_tokens = True,  # 添加special tokens， 也就是CLS和SEP
                                max_length = length,           # 设定最大文本长度
                                pad_to_max_length = True,   # pad到最大的长度  
                                return_tensors = 'pt'       # 返回的类型为pytorch tensor
                        )

    return sentence

tokenizer = BertTokenizer.from_pretrained('./bert-large-uncased/', do_lower_case=True)

puncts = ['.','!','?',',',';',':','[', ']', '{', '}', '(', ')', '\'', '\"']

author_list = {"EAP":0,"HPL":1, "MWS":2}

maxlen = 0
all_id  = []
data = []
labels  = []
with open(os.path.join(path,'data','train.csv'), 'r') as f:
    count =0
    reader = csv.reader(f)
    for row in tqdm(reader):
        if count == 0:
            count+=1
            continue
        # count +=1
        id = row[0]
        sentence = row[1]
        author = author_list[row[2]]
        # This process, however, afforded me no means of ascertaining the dimensions of my dungeon; as I might make its circuit, and return to the point whence I set out, without being aware of the fact; so perfectly uniform seemed the wall.
        sentence = sentence.lower()
        # this process, however, afforded me no means of ascertaining the dimensions of my dungeon; as i might make its circuit, and return to the point whence i set out, without being aware of the fact; so perfectly uniform seemed the wall.
        # sentence = remove_numbers(sentence)
        # this process, however, afforded me no means of ascertaining the dimensions of my dungeon; as i might make its circuit, and return to the point whence i set out, without being aware of the fact; so perfectly uniform seemed the wall.
        # sentence = remove_punctuation(sentence)
        # this process however afforded me no means of ascertaining the dimensions of my dungeon as i might make its circuit and return to the point whence i set out without being aware of the fact so perfectly uniform seemed the wall
        # sentence = remove_stopwords(sentence)
        # process however afforded   means  ascertaining  dimensions   dungeon   might make  circuit  return   point whence  set  without  aware   fact  perfectly uniform seemed  wall
        sentence = remove_whitespace(sentence)
        # process however afforded means ascertaining dimensions dungeon might make circuit return point whence set without aware fact perfectly uniform seemed wall
        sentence = encode_fn(sentence,512)
        # tensor([[  101,  2832,  2174, 22891,  2965,  2004, ... 0,     0,     0]])

        all_id.append(id)
        labels.append(author)
        data.append(sentence)
        # if count==2:
        #     break
all_data = torch.cat(data, dim=0)
labels = torch.tensor(labels)
dataset = (all_data, labels, all_id)

def test_preprocess():
    sentence = "I like you very much, But I like your         100 CAT too!!!"
    print(sentence)
    sentence = sentence.lower()
    print(sentence)
    sentence = remove_numbers(sentence)
    print(sentence)
    sentence = remove_punctuation(sentence)
    print(sentence)
    sentence = remove_stopwords(sentence)
    print(sentence)
    sentence = remove_whitespace(sentence)
    print(sentence)
    sentence = encode_fn(sentence,512)
    print(sentence)

saved_name = "/data/processed_train"
pickle.dump(dataset,open(path+saved_name,"wb"))
test = pickle.load(open(path+saved_name,"rb"))
print (test[0])
