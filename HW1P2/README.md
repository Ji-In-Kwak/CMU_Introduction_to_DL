# Homework 1 Part 2


### [Kaggle]

https://www.kaggle.com/competitions/11-785-f22-hw1p2/overview

### [Dataset]

kaggle competitions download -c 11-785-f22-hw1p2

### [Implementation]

To save the best model parameters and the prediction result file, make a dictionary. 
```
mkdir exp_structure_v2
```

Then, run the HW1P2_structure_exp_v2.ipynb file to preprocess the data, contruct the model architecture, and train the model with dataset. After finish training, the prediction values for the test set will saved as a ```.csv``` format in ```./exp_structure_v2/submission_largemodel_1.csv```.  

### [Model Experiments]
I have conducted several experiments regarding to the hyperparameters such as model architecture, number of contexts, and scheduler parameters. We can see that the best model approach has decreasing validation loss for 50 epochs

<img width="710" alt="wandb performance screenshot" src="https://user-images.githubusercontent.com/74504090/195491776-a4dc6e93-d908-41a1-8279-4845af977182.PNG">
