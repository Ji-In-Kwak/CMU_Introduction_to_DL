# HW2P2 Face Recognition and Verification

## Network
EfficientNet_v2 is used to implement the face recognition and verification in this homework. 
The `effnet.py` includes the code of model structures and we need to run `HW2P2_effnet_v2.ipynb` to train the model and test the results.


## Hyperparameters
Several hyperparameters are determined before the importing the model structure and training the model. The parameter values are set in the `config` directory. The word mentioned inside the bracket is the abbreviation for each parameters.
- batch_size : 64
- learning rate (lr) : 0.2
- scheduler : cosine annealing
- T_max (lr_stepsize) : batch_size * epochs 
- number of epochs (epochs) : 100
- optimizer : SGD
- weight_decay : 1e-5
- smoothing : 0.1
- dropout : 0.15


## Run the code
You can just run this code for the implementation. What you should do is 1) to set the `config` with the desired hyperparameters, 2) login the wandb with your key ID, and 3) set the name of the root directory path where you want to save your result model weights and csv files. You can also change your experiment name with the variable `exp_name`. The jupyter notebook code will load the model from the python file and run the training step. 


