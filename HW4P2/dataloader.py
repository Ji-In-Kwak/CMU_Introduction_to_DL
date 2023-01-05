import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, VOCAB, VOCAB_MAP, partition='train'): 
        '''
        Initializes the dataset.

        INPUTS: denote the dataset is 'train' or 'val'
        '''

        # Load the directory and all files in them
        if partition == 'train':
            self.mfcc_dir = 'data/hw4p2/train-clean-100/mfcc'#TODO
            self.transcript_dir = 'data/hw4p2/train-clean-100/transcript/raw'#TODO
        else:
            self.mfcc_dir = 'data/hw4p2/dev-clean/mfcc'#TODO
            self.transcript_dir = 'data/hw4p2/dev-clean/transcript/raw'#TODO
            

        self.mfcc_files = sorted(os.listdir(self.mfcc_dir)) #TODO
        self.transcript_files = sorted(os.listdir(self.transcript_dir)) #TODO

        self.vocab = VOCAB
        self.vocab_map = VOCAB_MAP

        # WHAT SHOULD THE LENGTH OF THE DATASET BE?
        assert len(self.mfcc_files) == len(self.transcript_files)
        self.length = len(self.mfcc_files)
        
       

    def __len__(self):
        
        '''
        output: length of mfcc
        '''
        return self.length

    def __getitem__(self, ind):
        '''
        TODO: RETURN THE MFCC COEFFICIENTS AND ITS CORRESPONDING LABELS

        If you didn't do the loading and processing of the data in __init__,
        do that here.

        Once done, return a tuple of features and labels.
        '''
        
        mfcc = np.load(os.path.join(self.mfcc_dir, self.mfcc_files[ind]))
        transcript = np.load(os.path.join(self.transcript_dir, self.transcript_files[ind]))
        transcript = np.array([self.vocab_map[t] for t in transcript])
        
        return mfcc, transcript


    def collate_fn(self,batch):
        '''
        TODO:
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish. 
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features, 
            and lengths of labels.
        '''

        batch_mfcc, batch_transcript = [], []
        for b in batch:
            batch_mfcc.append(torch.Tensor(b[0]))
            batch_transcript.append(torch.LongTensor(b[1]))

        
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True, padding_value=0) 
        lengths_mfcc = [batch_mfcc[i].shape[0] for i in range(len(batch_mfcc))] 

        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True, padding_value=0)
        lengths_transcript = [batch_transcript[i].shape[0] for i in range(len(batch_transcript))]

        # You may apply some transformation, Time and Frequency masking, here in the collate function;
        
        # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)
    
    
# Test Dataloader
class AudioDatasetTest(torch.utils.data.Dataset):
    def __init__(self):
        self.mfcc_dir = 'data/hw4p2/test-clean/mfcc'
        
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))
        self.length = len(self.mfcc_files)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, ind):
        mfcc = np.load(os.path.join(self.mfcc_dir, self.mfcc_files[ind]))
        return mfcc
        
    def collate_fn(self, batch):
        batch_mfcc = []
        for b in batch:
            batch_mfcc.append(torch.Tensor(b))
            
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True, padding_value=0)
        lengths_mfcc = [batch_mfcc[i].shape[0] for i in range(len(batch_mfcc))]
        
        return batch_mfcc_pad, torch.tensor(lengths_mfcc) 