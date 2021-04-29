
# -*- coding: utf-8 -*-
import sys
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import nn
import pickle
from tqdm import  tqdm
from bert_lstm_model import bert_lstm
from common import save_checkpoint, flat_accuracy

path = os.getcwd()
print (path) 
USE_CUDA = torch.cuda.is_available()
if(USE_CUDA):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')
device = torch.device("cuda" if USE_CUDA else "cpu")

# get preprocessed data
data = pickle.load(open(path+"/data/processed_train","rb"))
all_input_ids = data[0]
labels= data[1]

# Split data into train and validation
dataset = TensorDataset(all_input_ids, labels)
train_size = int(0.90 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create train and validation dataloaders
batch_size = 5
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True,drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False,drop_last=True)

output_size = 3     # three classes
hidden_dim = 384   #768/2
n_layers = 2
bidirectional = True  #这里为True，为双向LSTM
model_kind = "bert_lstm"

# move model to GPU, if available
model = bert_lstm(hidden_dim, output_size,n_layers, bidirectional)
model.to(device)

lr=5e-5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# training params
epochs = 50
clip=5 # gradient clipping
 



# train for some number of epochs
for epoch in range(epochs):
    total_loss, total_val_loss = 0, 0
    total_eval_accuracy = 0
    total_train_accuracy = 0
    # initialize hidden state
    h = model.init_hidden(batch_size)
    model.train()
    # batch loop
    for inputs, labels in tqdm(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        h = tuple([each.data for each in h])
        model.zero_grad()
        # intputs [b, 512]
        output= model(inputs, h)  # output [b,1]
        # print(output.shape)
        loss = criterion(output, labels.long())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        output = output.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        total_train_accuracy += flat_accuracy(output, labels)

    model.eval()
    with torch.no_grad():
        val_h = model.init_hidden(batch_size)
        for inputs, labels in tqdm(val_dataloader):
            val_h = tuple([each.data for each in val_h])
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs, val_h)
            val_loss = criterion(output, labels.long())
            total_val_loss += val_loss
            # prediction
            pred = output.argmax(dim=1)     
            
            output = output.detach().cpu().numpy()
            labels = labels.cpu().numpy()
            total_eval_accuracy += flat_accuracy(output, labels)

        model.train()

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
    save_checkpoint(model_kind,model,batch_size, lr, epoch, avg_val_accuracy,avg_val_loss)


