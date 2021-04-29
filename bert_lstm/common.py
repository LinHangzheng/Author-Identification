import os
import pickle
import random
import numpy as np
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

def save_checkpoint(model_kind, model, batch, lr, epoch, avg_val_accuracy,avg_val_loss):
    path = os.getcwd()
    save_path = path + f"/checkpoints/{model_kind}/batch_{batch}_lr_{lr}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)  
    pickle.dump(model,open(save_path+"/bert_epoch_"+str(epoch+1),"wb"))
    pickle.dump(avg_val_accuracy,open(save_path+f"/epoch_{epoch+1}_acc_{avg_val_accuracy:.2f}_loss_{avg_val_loss:.2f}", 
                            "wb"))

                    