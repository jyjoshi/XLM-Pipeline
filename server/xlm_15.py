#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np

#import evaluate


#metric = evaluate.load("accuracy")


# In[7]:
'''

with open("Finetuned_Models/Spanish/Haternet/labeled_corpus_6K.txt", encoding='utf-8') as f:

    lines = f.readlines()
    
ds = [[y, int(z[1:2])] for _, y, z in [line.split("-,-") for line in lines][1:]]

texts = [i[0] for i in ds]
labels = [i[1] for i in ds]

'''


# In[15]:
import csv

with open("Finetuned_Models/English/Hateval/hateval2019_en_train.csv", encoding='utf-8') as f:
	ds_train = csv.reader(f)
	ds_train = [[text, label] for _, text, label, _, _ in ds_train]

with open("Finetuned_Models/English/Hateval/hateval2019_en_test.csv", encoding='utf-8') as f:
	ds_test = csv.reader(f)
	ds_test = [[text, label] for _, text, label, _, _ in ds_test]

train_texts = [i[0] for i in ds_train[1:]]
train_labels = [int(i[1]) for i in ds_train[1:]]


test_texts = [i[0] for i in ds_test[1:]]
test_labels = [int(i[1]) for i in ds_test[1:]]

# In[9]:


from sklearn.model_selection import train_test_split
# train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=.1)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)


# In[13]:


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-mlm-xnli15-1024', download_mode="force_redownload")


# In[16]:


train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)


# In[17]:


import torch

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)




# In[ ]:


from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results/xlm_15',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    #evaluation_strategy="epoch",
)

model = AutoModelForSequenceClassification.from_pretrained("xlm-mlm-xnli15-1024")


trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
 #   compute_metrics=compute_metrics,
    tokenizer=tokenizer   
)

trainer.train()
trainer.save_model("FineTuned_Model/xlm_15")

# In[ ]:




