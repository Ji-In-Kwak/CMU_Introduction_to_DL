import wandb

import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torchaudio.transforms as tat

# from sklearn.metrics import accuracy_score
import gc

import zipfile
import pandas as pd
from tqdm import tqdm
import os
import datetime

# imports for decoding and distance calculation
import ctcdecode
import Levenshtein
from ctcdecode import CTCBeamDecoder

import warnings
warnings.filterwarnings('ignore')

from dataset import AudioDataset, AudioDatasetTest
from model import Network
from levenshtein_dist import calculate_levenshtein
from trainer import train_step, evaluate, make_output


# os.environ["CUDA_VISIBLE_DEVICES"]=
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

config = {
        "exp_dir": "exp_LR",
        "exp_name": "cosine_100_1",
        "batch_size": 64,
        "beam_width" : 2,
        "lr" : 2e-3,
        "epochs" : 100,
        "weight_decay" : 0.0001,
        "dropout" : 0.3,
        "CNN_dim": 128,
        "RNN_model" : "LSTM",
        "RNN_n_layers" : 5,
        "LR" : 'Cosine',
        "LR_factor": 100
        }


def predict(test_loader, model, decoder, LABELS):
    pred_all = []
    
    for data in tqdm(test_loader):
        x, lx = data
        x = x.to(device)
        
        with torch.inference_mode():
            out, out_lengths = model(x, lx)
#             out = out.permute(1, 0, 2)
            
        pred = make_output(out, out_lengths, decoder, LABELS, config['batch_size'])
        pred_all.extend(pred)
        
    return pred_all



def main():
    wandb.login(key="db668044fcf11bc7352d5c79ac0deb75ac86a16b")

    
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('results/'+config['exp_dir']):
        os.mkdir('results/'+config['exp_dir'])

    run = wandb.init(
        name = config['exp_dir'], ## Wandb creates random run names if you skip this field
        reinit = True, ### Allows reinitalizing runs when you re-run this cell
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project = "hw3p2-ablations", ### Project should be created in your wandb account 
        config = config ### Wandb Config for your run
    )


    ## Dataset Loading
    gc.collect()
    
    CMUdict_ARPAbet = {
    "" : " ", # BLANK TOKEN
    "[SIL]": "-", "NG": "G", "F" : "f", "M" : "m", "AE": "@", 
    "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W", 
    "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R", 
    "HH"   : "h", "Z" : "z", "K" : "k", "CH": "C", "W" : "w", 
    "EY"   : "e", "ZH": "Z", "T" : "t", "EH": "E", "Y" : "y", 
    "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D", 
    "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O", 
    "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
    "[SOS]": "[SOS]", "[EOS]": "[EOS]"}

    CMUdict = list(CMUdict_ARPAbet.keys())
    ARPAbet = list(CMUdict_ARPAbet.values())


    PHONEMES = CMUdict
    mapping = CMUdict_ARPAbet
    LABELS = ARPAbet
    
    BATCH_SIZE = config['batch_size']
    OUT_SIZE = len(LABELS)
    
    # Create objects for the dataset class
    train_data = AudioDataset(partition='train') #TODO
    val_data = AudioDataset(partition='val') # TODO : You can either use the same class with some modifications or make a new one :)
    test_data = AudioDatasetTest() #TODO

    # Do NOT forget to pass in the collate function as parameter while creating the dataloader
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=train_data.collate_fn)#TODO
    val_loader = DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=val_data.collate_fn) #TODO
    test_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE, collate_fn=test_data.collate_fn) #TODO

    print("Batch size: ", BATCH_SIZE)
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
    print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

    ## Sanity check
    for data in train_loader:
        x, y, lx, ly = data
        print(x.shape, y.shape, lx.shape, ly.shape)
        break 

    ## Model 
    model = Network(OUT_SIZE, config['RNN_n_layers'], config['dropout'], cnn_n_dims=config['CNN_dim']).to(device)
    torch.cuda.empty_cache()
    summary(model, x.to(device), lx) # x and lx come from the sanity check above :)

    # parallel_net = nn.DataParallel(model)#, device_ids = [0,1,2,3])

    ## Train
    criterion = nn.CTCLoss()
    optimizer =  torch.optim.AdamW(model.parameters(), lr = config['lr'], weight_decay=config['weight_decay'])
    decoder = ctcdecode.CTCBeamDecoder(LABELS, beam_width=config['beam_width'], log_probs_input=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = torch.cuda.amp.GradScaler()
    
    last_epoch_completed = 0
    start = last_epoch_completed
    end = config['epochs']
    best_val_dist = float("inf") # if you're restarting from some checkpoint, use what you saw there.
    best_val_loss = float("inf")
    dist_freq = 1

    torch.cuda.empty_cache()
    gc.collect()
    
    for epoch in range(config["epochs"]):
        
        curr_lr = float(optimizer.param_groups[0]['lr'])

        # one training step
        # one validation step (if you want)
        train_loss = train_step(train_loader, model, optimizer, criterion, scheduler, scaler, device)
        
        print("\nEpoch {}/{}: \nTrain Loss {:.04f}\t Learning Rate {:.04f}".format(
            epoch + 1,
            config['epochs'],
            train_loss,
            curr_lr))
        
        if epoch % 5 == 0:
            val_loss, val_dist = evaluate(val_loader, model, criterion, decoder, device, LABELS)
            print("Val Loss {:.04f}\t Val Dist {:.04f}".format(val_loss, val_dist))
        
        curr_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        # HINT: Calculating levenshtein distance takes a long time. Do you need to do it every epoch?
        # Does the training step even need it? 

        # Where you have your scheduler.step depends on the scheduler you use.
        
        # Use the below code to save models
        if val_dist < best_val_dist:
            print("Saving model")
            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_dist': val_dist, 
                        'epoch': epoch}, './results/{}/{}_checkpoint.pth'.format(config['exp_dir'], config['exp_name']))
            best_val_loss = val_loss
            wandb.save('checkpoint.pth')

        # You may want to log some hyperparameters and results on wandb
        wandb.log({"train_loss":train_loss, 'validation_loss': val_loss, "validation_dist": val_dist, "learning_rate": curr_lr})

    run.finish()
    
    
    decoder_test = CTCBeamDecoder(LABELS, beam_width=50, log_probs_input=True)

    model = model.to(device)
    model.load_state_dict(torch.load('./results/{}/{}_checkpoint.pth'.format(config['exp_dir'], config['exp_name']))['model_state_dict'])
    
    torch.cuda.empty_cache()
    predictions = predict(test_loader, model, decoder_test, LABELS)
    
    df = pd.read_csv('hw3p2/test-clean/transcript/random_submission.csv')
    df.label = predictions

    df.to_csv('results/{}/{}_submission.csv'.format(config['exp_dir'], config['exp_name']), index = False)



if __name__ == "__main__":
    main()


