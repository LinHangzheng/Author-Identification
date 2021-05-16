import os
import pickle
import random
import numpy as np
import logging
import torch
from sklearn.metrics import f1_score, accuracy_score

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def flat_accuracy(preds, labels):
    
    """A function for calculating accuracy scores"""
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)

def save_checkpoint(args, model, epoch, avg_val_accuracy,avg_val_loss):
    path = os.getcwd()
    logging.basicConfig(level=logging.INFO, filename=args.exp_name+'/evaluate.log',filemode='w')
    pickle.dump(model,open(args.exp_name+"/models/model"+str(epoch+1),"wb"))
    logging.info(
        f'Epoch     : {epoch+1}'
        f'Average Validation Accuracy: {avg_val_accuracy:.2g}'
        f'Average Validation loss: {avg_val_loss:.2g}'
    )


                    