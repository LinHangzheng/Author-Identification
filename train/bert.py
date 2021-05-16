# -*- coding: utf-8 -*-
import os
import numpy as np
import random
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
import pickle
from tqdm import  tqdm
# settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

epochs = 30
batch_size = 20

data = pickle.load(open("../data/processed_train","rb"))
all_input_ids = data[0]
labels= data[1]

# Split data into train and validation
dataset = TensorDataset(all_input_ids, labels)
train_size = int(0.90 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create train and validation dataloaders
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

# Load the pretrained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3, output_attentions=False, output_hidden_states=False)
model.to(device)

lr = 2e-5
# create optimizer and learning rate schedule
optimizer = AdamW(model.parameters(), lr=lr)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


def flat_accuracy(preds, labels):
    
    """A function for calculating accuracy scores"""
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)

def save_checkpoint(model, epoch, avg_val_accuracy,avg_val_loss):
    pickle.dump(model,open("../checkpoints/bert_epoch_"+str(epoch+1),"wb"))
    pickle.dump(avg_val_accuracy,open("../checkpoints/epoch_"+
                              str(epoch+1)+"acc_"+str(round(avg_val_accuracy,2))+
                              "_loss_"+str(round(avg_val_loss,2)),"wb"))

load_check = 0
# model = pickle.load(open("/content/drive/MyDrive/ECE412/checkpoint/bert_epoch_"+str(load_check),"rb"))

for epoch in range(load_check, epochs):
    model.train()
    total_loss, total_val_loss = 0, 0
    total_eval_accuracy = 0
    total_train_accuracy = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        model.zero_grad()
        outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device), labels=batch[1].to(device))
        logits = outputs.logits
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        # clipping gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step() 
        scheduler.step()

        logits = logits.detach().cpu().numpy()
        label_ids = batch[1].cpu().numpy()
        total_train_accuracy += flat_accuracy(logits, label_ids)

    model.eval()
    for i, batch in enumerate(tqdm(val_dataloader)):
        with torch.no_grad():
            outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device), labels=batch[1].to(device))
            logits = outputs.logits
            loss = outputs.loss
            total_loss += loss.item()
            total_val_loss += loss.item()
            
            logits = logits.detach().cpu().numpy()
            label_ids = batch[1].cpu().numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
    
    avg_train_loss = total_loss / len(train_dataloader)
    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    print(f'Epoch     : {epoch+1}')
    print(f'Train loss     : {avg_train_loss}')
    print(f'Train Accuracy: {avg_train_accuracy:.2f}')
    print(f'Validation loss: {avg_val_loss}')
    print(f'Validation Accuracy: {avg_val_accuracy:.2f}')
    print('\n')
    save_checkpoint(model,epoch, avg_val_accuracy,avg_val_loss)
    