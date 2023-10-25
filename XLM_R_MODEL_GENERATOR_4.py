#!/usr/bin/env python
# coding: utf-8

# In[22]:


# get_ipython().system('pip install transformers')
# get_ipython().system('pip install torch torchvision')
# get_ipython().system('pip install csv-reader')
# get_ipython().system('pip install -U scikit-learn')


# In[23]:


'''from pathlib import Path

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels

train_texts, train_labels = read_imdb_split('aclImdb/train')
test_texts, test_labels = read_imdb_split('aclImdb/test')'''


# In[3]:


# import pandas as pd

# ds_train = pd.read_csv("Hateval/hateval2019_en_train.csv")
# ds_test = pd.read_csv("Hateval/hateval2019_en_test.csv")

# x = ds_train['HS']
# train_labels = [i for i in x]
# train_texts = [i for i in ds_train['text']]

# test_labels = [i for i in ds_test['HS']]
# test_texts = [i for i in ds_test['text']]


# In[4]:


import csv


# In[5]:


# Getting all English Data
en1_texts = []
en1_labels=[]
en2_texts = []
en2_labels = []
with open("CleanData/Finetuned_Models/English/Hateval/data.csv", encoding='utf-8') as f:
    ds = csv.reader(f)
    ds = [[text, label] for text, label in ds]
    for i in range(len(ds)):
        en1_texts.append(ds[i][0])
        en1_labels.append(ds[i][1])
        
with open("CleanData/Finetuned_Models/English/HASOC track at FIRE 2019_ Hate Speech and Offensive Content Identification in Indo-European Languages/english_dataset.tsv", encoding='utf-8') as f:
    for line in f:
        x = line.split("\t")
        en2_texts.append(x[1])
        en2_labels.append(x[2])
        

en1_labels = [int(i) for i in en1_labels[1:]]
en2_labels = [int(i) for i in en2_labels[1:]]
en1_texts = en1_texts[1:]
en2_texts = en2_texts[1:]


# In[6]:


# Getting all Spanish Data
es1_texts = []
es1_labels = []
es2_texts = []
es2_labels = []
with open ("CleanData/Finetuned_Models/Spanish/Haternet/data.csv", encoding='utf-8') as f:
    ds = csv.reader(f)
    ds = [[text, label] for _, text, label in ds]
    for i in range(len(ds)):
        es1_labels.append(ds[i][1])
        es1_texts.append(ds[i][0])
        
with open ("CleanData/Finetuned_Models/Spanish/Hateval/data.csv", encoding='utf-8') as f:
    ds = csv.reader(f)
    ds = [[text, label] for _, text, label, _, _ in ds]
    for i in range(len(ds)):
        es2_labels.append(ds[i][1])
        es2_texts.append(ds[i][0])

es1_labels = [int(i) for i in es1_labels[1:]]
es1_texts = es1_texts[1:]
es2_labels = [int(i) for i in es2_labels[1:]]
es2_texts = es2_texts[1:]
    


# In[8]:


# Getting all Hindi Data 
hi1_texts = []
hi1_labels = []
hi2_texts = []
hi2_labels = []
with open ("CleanData/Finetuned_Models/Hindi/HASOC track at FIRE 2019_ Hate Speech and Offensive Content Identification in Indo-European Languages/data.csv", encoding='utf-8') as f:
    ds = csv.reader(f)
    ds = [[text, label] for _, text, label, _, _, _ in ds]
    for i in range(len(ds)):
        hi1_labels.append(ds[i][1])
        hi1_texts.append(ds[i][0])

        
with open ("CleanData/Finetuned_Models/Hindi/Hostile Post Detection in Hindi/data.csv", encoding='utf-8') as f:
    ds = csv.reader(f)
    ds = [[text, label] for _, text, label, _ in ds]
    for i in range(len(ds)):
        hi2_labels.append(ds[i][1])
        hi2_texts.append(ds[i][0])
        
hi1_labels = [int(i) for i in hi1_labels[1:]]
hi1_texts = hi1_texts[1:]
hi2_labels = [int(i) for i in hi2_labels[1:]]
hi2_texts = hi2_texts[1:]
    
    


# In[9]:


from sklearn.model_selection import train_test_split
# Make train, validation and test sets for all datasets individually before aggregating. 

en1_train_texts, en1_test_texts, en1_train_labels, en1_test_labels = train_test_split(en1_texts, en1_labels, test_size=.1)
en1_train_texts, en1_val_texts, en1_train_labels, en1_val_labels = train_test_split(en1_train_texts, en1_train_labels, test_size=.1)

en2_train_texts, en2_test_texts, en2_train_labels, en2_test_labels = train_test_split(en2_texts, en2_labels, test_size=.1)
en2_train_texts, en2_val_texts, en2_train_labels, en2_val_labels = train_test_split(en2_train_texts, en2_train_labels, test_size=.1)

es1_train_texts, es1_test_texts, es1_train_labels, es1_test_labels = train_test_split(es1_texts, es1_labels, test_size=.1)
es1_train_texts, es1_val_texts, es1_train_labels, es1_val_labels = train_test_split(es1_train_texts, es1_train_labels, test_size=.1)

es2_train_texts, es2_test_texts, es2_train_labels, es2_test_labels = train_test_split(es2_texts, es2_labels, test_size=.1)
es2_train_texts, es2_val_texts, es2_train_labels, es2_val_labels = train_test_split(es2_train_texts, es2_train_labels, test_size=.1)

hi1_train_texts, hi1_test_texts, hi1_train_labels, hi1_test_labels = train_test_split(hi1_texts, hi1_labels, test_size=.1)
hi1_train_texts, hi1_val_texts, hi1_train_labels, hi1_val_labels = train_test_split(hi1_train_texts, hi1_train_labels, test_size=.1)

hi2_train_texts, hi2_test_texts, hi2_train_labels, hi2_test_labels = train_test_split(hi2_texts, hi2_labels, test_size=.1)
hi2_train_texts, hi2_val_texts, hi2_train_labels, hi2_val_labels = train_test_split(hi2_train_texts, hi2_train_labels, test_size=.1)


# In[10]:


# Model Details:
# XLM_R trained on different subsets of languages English, Hindi and Spanish
# List of Models to train:
# 1. Model 1 -> FineTuned using only Hateval dataset (en-hateval)
# 2. Model 2 -> FineTuned using only HASOC dataset (en-hasoc)
# 3. Model 3 -> FineTuned using both English datasets (EN)
# 4. Model 4 -> FineTuned using both Spanish datasets (Haternet and Hateval) (ES)
# 5. Model 5 -> FineTuned using both Hindi datasets (HASOC and Hostile Post Detection in Hindi) (HI)
# 6. Model 6 -> FineTuned using Aggregation of English and Spanish datasets (EN-ES)
# 7. Model 7 -> FineTuned using Aggregation of English and Hindi datasets (HI-EN)
# 8. Model 8 -> FineTuned using Aggregation of Spanish and Hindi datasets (ES-HI)
# 9. Model 9 -> Finally an Aggregation of all the three datasets.(EN-ES-HI)


# In[11]:


# Aggregate English Data -> Model 3

en_train_texts = en1_train_texts 
en_val_texts = en1_val_texts
en_test_texts = en1_test_texts 
en_train_labels = en1_train_labels
en_val_labels = en1_val_labels
en_test_labels = en1_test_labels 


# In[12]:


# Aggregate Spanish Data -> Model 4

es_train_texts = es1_train_texts + es2_train_texts
es_val_texts = es1_val_texts + es2_val_texts
es_test_texts = es1_test_texts + es2_test_texts
es_train_labels = es1_train_labels + es2_train_labels
es_val_labels = es1_val_labels + es2_val_labels
es_test_labels = es1_test_labels + es2_test_labels


# In[13]:


# Aggregate Hindi Data -> Model 5

hi_train_texts = hi1_train_texts + hi2_train_texts
hi_val_texts = hi1_val_texts + hi2_val_texts
hi_test_texts = hi1_test_texts + hi2_test_texts
hi_train_labels = hi1_train_labels + hi2_train_labels
hi_val_labels = hi1_val_labels + hi2_val_labels
hi_test_labels = hi1_test_labels + hi2_test_labels


# In[14]:


# Aggregate EN-ES -> Model 6

en_es_train_texts = en_train_texts + es_train_texts
en_es_val_texts = en_val_texts + es_val_texts
en_es_test_texts = en_test_texts + es_test_texts
en_es_train_labels = en_train_labels + es_train_labels
en_es_val_labels = en_val_labels + es_val_labels
en_es_test_labels = en_test_labels + es_test_labels


# In[15]:


# Aggregate HI-ES -> Model 7

hi_es_train_texts = hi_train_texts + es_train_texts
hi_es_val_texts = hi_val_texts + es_val_texts
hi_es_test_texts = hi_test_texts + es_test_texts
hi_es_train_labels = hi_train_labels + es_train_labels
hi_es_val_labels = hi_val_labels + es_val_labels
hi_es_test_labels = hi_test_labels + es_test_labels


# In[16]:


# Aggregate HI-EN -> Model 8

hi_en_train_texts = hi_train_texts + en_train_texts
hi_en_val_texts = hi_val_texts + en_val_texts
hi_en_test_texts = hi_test_texts + en_test_texts
hi_en_train_labels = hi_train_labels + en_train_labels
hi_en_val_labels = hi_val_labels + en_val_labels
hi_en_test_labels = hi_test_labels + en_test_labels


# In[17]:


# Aggregate HI-EN-ES -> Model 9

all_train_texts = hi_en_train_texts + es_train_texts
all_val_texts = hi_en_val_texts + es_val_texts
all_test_texts = hi_en_test_texts + es_test_texts
all_train_labels = hi_en_train_labels + es_train_labels
all_val_labels = hi_en_val_labels + es_val_labels
all_test_labels = hi_en_test_labels + es_test_labels


# In[18]:


# Storing the ds in a key:value pair format so that easily accessible for functions

models = {
    "HI-EN-ES":{
        "train_texts":all_train_texts,
        "val_texts":all_val_texts,
        "test_texts":all_test_texts,
        "train_labels":all_train_labels,
        "val_labels":all_val_labels,
        "test_labels":all_test_labels,
    },
    "HI-EN":{
        "train_texts":hi_en_train_texts,
        "val_texts":hi_en_val_texts,
        "test_texts":hi_en_test_texts,
        "train_labels":hi_en_train_labels,
        "val_labels":hi_en_val_labels,
        "test_labels":hi_en_test_labels,
    },
    "HI-ES":{
        "train_texts":hi_es_train_texts,
        "val_texts":hi_es_val_texts,
        "test_texts":hi_es_test_texts,
        "train_labels":hi_es_train_labels,
        "val_labels":hi_es_val_labels,
        "test_labels":hi_es_test_labels,
    },
    "EN-ES":{
        "train_texts":en_es_train_texts,
        "val_texts":en_es_val_texts,
        "test_texts":en_es_test_texts,
        "train_labels":en_es_train_labels,
        "val_labels":en_es_val_labels,
        "test_labels":en_es_test_labels,
    },
    "HI":{
        "train_texts":hi_train_texts,
        "val_texts":hi_val_texts,
        "test_texts":hi_test_texts,
        "train_labels":hi_train_labels,
        "val_labels":hi_val_labels,
        "test_labels":hi_test_labels,
    },
    "ES":{
        "train_texts":es_train_texts,
        "val_texts":es_val_texts,
        "test_texts":es_test_texts,
        "train_labels":es_train_labels,
        "val_labels":es_val_labels,
        "test_labels":es_test_labels,
    },
    "EN":{
        "train_texts":en_train_texts,
        "val_texts":en_val_texts,
        "test_texts":en_test_texts,
        "train_labels":en_train_labels,
        "val_labels":en_val_labels,
        "test_labels":en_test_labels,
    },
    "en1":{
        "train_texts":en1_train_texts,
        "val_texts":en1_val_texts,
        "test_texts":en1_test_texts,
        "train_labels":en1_train_labels,
        "val_labels":en1_val_labels,
        "test_labels":en1_test_labels,
    },
#     "en2":{
#         "train_texts":en2_train_texts,
#         "val_texts":en2_val_texts,
#         "test_texts":en2_test_texts,
#         "train_labels":en2_train_labels,
#         "val_labels":en2_val_labels,
#         "test_labels":en2_test_labels,
#     }
}


# In[19]:


import torch
torch.cuda.empty_cache()
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# In[20]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base")


# In[30]:


# Get all the relevant values
# train_texts = ds['train_texts']
# val_texts = ds['val_texts']
# test_texts = ds['test_texts']
# train_labels = ds['train_labels']
# val_labels = ds['val_labels']
# test_labels = ds['test_labels']

# train_texts = train_texts
# val_texts = val_texts
# test_texts = test_texts
# train_labels = train_labels
# val_labels = val_labels
# test_labels = test_labels
for key in models.keys():
    train_texts = models[key]["train_texts"]
    val_texts = models[key]["val_texts"]
    test_texts = models[key]["test_texts"]
    train_labels = models[key]["train_labels"]
    val_labels = models[key]["val_labels"]
    test_labels = models[key]["test_labels"]

    #Tokenize the input texts and labels
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    # Structure the encodings in the right format so that they can be provided to the trainer. 
    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, val_labels)
    test_dataset = Dataset(test_encodings, test_labels)



    training_args = TrainingArguments(

        output_dir='./results/xlm_r/' + key,          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs/xlm_r/' + key,            # directory for storing logs
        logging_steps=10,
    )



    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    trainer.train()
    trainer.save_model("./results/xlm_r/" + key )

# In[ ]:




