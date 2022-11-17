import os
import sys
import numpy as np
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


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
    

class AudioDataset(torch.utils.data.Dataset):

    # For this homework, we give you full flexibility to design your data set class.
    # Hint: The data from HW1 is very similar to this HW

    #TODO
    def __init__(self, partition='train'): 
        '''
        Initializes the dataset.

        INPUTS: What inputs do you need here?
        '''

        # Load the directory and all files in them

        if partition == 'train':
            self.mfcc_dir = 'hw3p2/train-clean-100/mfcc'#TODO
            self.transcript_dir = 'hw3p2/train-clean-100/transcript/raw'#TODO
        else:
            self.mfcc_dir = 'hw3p2/dev-clean/mfcc'#TODO
            self.transcript_dir = 'hw3p2/dev-clean/transcript/raw'#TODO
            

        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))[:] #TODO
        self.transcript_files = sorted(os.listdir(self.transcript_dir))[:] #TODO

        self.PHONEMES = PHONEMES

        #TODO
        # WHAT SHOULD THE LENGTH OF THE DATASET BE?
        assert len(self.mfcc_files) == len(self.transcript_files)
        self.length = len(self.mfcc_files)
        
        #TODO
        # HOW CAN WE REPRESENT PHONEMES? CAN WE CREATE A MAPPING FOR THEM?
        # HINT: TENSORS CANNOT STORE NON-NUMERICAL VALUES OR STRINGS
        
        enum = enumerate(PHONEMES)
        self.PHON_dict = dict((j, i) for i,j in enum)


        #TODO
        # CREATE AN ARRAY OF ALL FEATUERS AND LABELS
        # WHAT NORMALIZATION TECHNIQUE DID YOU USE IN HW1? CAN WE USE IT HERE?
        '''
        You may decide to do this in __getitem__ if you wish.
        However, doing this here will make the __init__ function take the load of
        loading the data, and shift it away from training.
        '''

    def __len__(self):
        
        '''
        TODO: What do we return here?
        '''
        return self.length

    def __getitem__(self, ind):
        '''
        TODO: RETURN THE MFCC COEFFICIENTS AND ITS CORRESPONDING LABELS

        If you didn't do the loading and processing of the data in __init__,
        do that here.

        Once done, return a tuple of features and labels.
        '''
        
        mfcc = np.load(os.path.join(self.mfcc_dir, self.mfcc_files[ind])) # TODO
        transcript = np.load(os.path.join(self.transcript_dir, self.transcript_files[ind])) # TODO
        transcript = np.array([self.PHON_dict[t] for t in transcript])
        
        return mfcc, transcript


    def collate_fn(self,batch):
        
        batch_mfcc, batch_transcript = [], []
        for b in batch:
            batch_mfcc.append(torch.Tensor(b[0]))
            batch_transcript.append(torch.Tensor(b[1]))


        # HINT: CHECK OUT -> pad_sequence (imported above)
        # Also be sure to check the input format (batch_first)
        
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True, padding_value=0) # TODO
        lengths_mfcc = [batch_mfcc[i].shape[0] for i in range(len(batch_mfcc))] # TODO ??

        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True, padding_value=0) # TODO
        lengths_transcript = [batch_transcript[i].shape[0] for i in range(len(batch_transcript))] # TODO ??
        
        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)

       
class AudioDatasetTest(torch.utils.data.Dataset):
    def __init__(self):
        self.mfcc_dir = 'hw3p2/test-clean/mfcc'
        
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