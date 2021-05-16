# -*- coding: utf-8 -*-
import sys
import argparse
import os
import torch
import logging
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import nn
import pickle
from torch.utils.tensorboard import SummaryWriter
from tqdm import  tqdm
from bert_lstm_model import bert_lstm
from common import save_checkpoint, flat_accuracy
# os.environ['CUDA_VISIBLE_DEVICES']='0'
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
if USE_CUDA:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def main(args):
    _init_(args)
    train(args)

def _init_(args):
    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)
    if not os.path.exists(args.exp_name + '/models'):
        os.makedirs(args.exp_name + '/models')
    if not os.path.exists(args.exp_name + '/results'):
        os.makedirs(args.exp_name + '/results')


# get preprocessed data
def load_data():
    path = os.getcwd()
    data = pickle.load(open(path+"/data/processed_train","rb"))
    all_input_ids = data[0]
    labels= data[1]
    return all_input_ids,labels

def create_dataset(all_input_ids,labels):
    # Split data into train and validation
    dataset = TensorDataset(all_input_ids, labels)
    train_size = int(0.90 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

# Create train and validation dataloaders
def create_dataloader(args,train_dataset,val_dataset):
    batch_size = args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True,drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False,drop_last=True)
    return train_dataloader, val_dataloader



#   move model to GPU, if available
def build_model(args):
    output_size = 3     # three classes
    if args.model == "./bert-large-uncased":
        hidden_dim = 512    #1024/2
    else:
        hidden_dim = 384   #768/2
    n_layers = 2
    model = bert_lstm(args.model, hidden_dim, output_size,n_layers, args)
    model.to(device)
    return model

def build_criterion():
    criterion = nn.CrossEntropyLoss()
    return criterion

def build_optim(args,model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return optimizer


def train(args):
    logging.basicConfig(level=logging.INFO, filename=args.exp_name+'/train.log',filemode='w')
    # train for some number of epochs
    writer = SummaryWriter(log_dir=args.exp_name + '/summary')

    # build the Bert model
    model = build_model(args)

    # build the optimizier
    optimizer = build_optim(args,model)

    # build the criterion
    criterion = build_criterion()
    
    # load data
    all_input_ids, labels = load_data()

    # create dataset
    train_dataset, val_dataset = create_dataset(all_input_ids,labels)
    
    # create the data loader
    train_loader, test_loader = create_dataloader(args,train_dataset,val_dataset)

    
    for epoch in range(args.epoch):
        total_loss, total_val_loss = 0, 0
        total_eval_accuracy = 0
        total_train_accuracy = 0
        # initialize hidden state
        h = model.init_hidden(args.batch_size,USE_CUDA)
        
        model.train()
        # batch loop
        for inputs, labels in tqdm(train_loader):
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
            val_h = model.init_hidden(args.batch_size)
            for inputs, labels in tqdm(test_loader):
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

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(test_loader)
        avg_val_accuracy = total_eval_accuracy / len(test_loader)
        avg_train_accuracy = total_train_accuracy / len(train_loader)
        logging.info(
            f'Epoch     : {epoch+1}'
            f'Train loss     : {avg_train_loss}'
            f'Train Accuracy: {avg_train_accuracy:.2f}'
            f'Validation loss: {avg_val_loss}'
            f'Validation Accuracy: {avg_val_accuracy:.2f}'
        )
        save_checkpoint(args, model,epoch, avg_val_accuracy,avg_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Author identification')
    parser.add_argument('--batch_size', type=int, default=3, metavar='batch_size',
                            help='Size of batch')
    parser.add_argument('--clip', type=int, default=5, metavar='clip_number',
                            help='value of clip')
    parser.add_argument('--epoch', type=int, default=50, metavar='epoch_number',
                            help='epoch number')
    parser.add_argument('--model', type=str, default='./bert-large-uncased', metavar='N',
                            choices=['bert-base-uncased','./bert-large-uncased'],
                            help='Model to use, [bert-large-uncased, bert-base-uncased]')
    parser.add_argument('--lr', type=float, default=0.00002, metavar='LR',
                            help='learning rate (default: 0.00001, 0.1 if using sgd)')
    parser.add_argument('--use_biLSTM', type=bool, default=True,
                        help='Use biLSTM of LSTM')
    parser.add_argument('--exp_name', type=str, 
                        default='./checkpoints/bert_large',
                        help='Name of the experiment')
    args = parser.parse_args()
    
    main(args)


