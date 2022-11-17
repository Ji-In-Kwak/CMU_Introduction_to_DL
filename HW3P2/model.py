import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence



torch.cuda.empty_cache()

class Network(nn.Module):

    def __init__(self, out_size, rnn_n_layers, dropout, cnn_n_dims=64, rnn_n_dims=256):

        super(Network, self).__init__()

        # Adding some sort of embedding layer or feature extractor might help performance.
        self.embedding = nn.Sequential(nn.Conv1d(in_channels=15, out_channels=64, kernel_size=5, padding=2, stride=2),
                                       nn.Dropout(dropout),
                                       nn.GELU(),
                                       nn.Conv1d(in_channels=64, out_channels=cnn_n_dims, kernel_size=5, padding=2, stride=2),
                                       nn.Dropout(dropout),
                                       nn.GELU()
                                      )
        
        # TODO : look up the documentation. You might need to pass some additional parameters.
        self.lstm = nn.LSTM(input_size = cnn_n_dims, hidden_size = rnn_n_dims, num_layers = rnn_n_layers, bidirectional=True, dropout=dropout) 
       
        self.classification = nn.Sequential(nn.Linear(256*2, 1024),
                                            nn.Dropout(dropout),
                                            nn.GELU(),
                                            nn.Linear(1024, 512),
                                            nn.Dropout(dropout), 
                                            nn.GELU(),
                                            nn.Linear(512, out_size)
            #TODO: Linear layer with in_features from the lstm module above and out_features = OUT_SIZE
        )

        
        self.logSoftmax = nn.LogSoftmax(dim=2) #TODO: Apply a log softmax here. Which dimension would apply it on ?

    def forward(self, x, lx):
        #TODO
        # The forward function takes 2 parameter inputs here. Why?
        # Refer to the handout for hints
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        # print(max(x[:, 0, :]))
        max_len = x.shape[1]
        lx = torch.clamp(lx, max=max_len)
        # lx = torch.Tensor([int((lx[i].item()-5+2*2)//2) for i in range(len(lx))])
        # print(lx[:5])
        packed = pack_padded_sequence(x, lx, batch_first=True, enforce_sorted=False)
        out, (hn, cn) = self.lstm(packed)
        seq_unpacked, lens_unpacked = pad_packed_sequence(out, batch_first=True)
        seq_unpacked = self.classification(seq_unpacked)
        seq_unpacked = self.logSoftmax(seq_unpacked)
        
        return seq_unpacked, lens_unpacked