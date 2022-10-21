import numpy as np


class Identity:
    
    def forward(self, Z):
    
        self.A = Z
        
        return self.A
    
    def backward(self):
    
        dAdZ = np.ones(self.A.shape, dtype="f")
        
        return dAdZ


class Sigmoid:
    
    def forward(self, Z):
    
        self.A = 1/(1+np.exp(-Z)) # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = self.A * (1 - self.A) # TODO
        
        return dAdZ


class Tanh:
    
    def forward(self, Z):
    
        self.A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z)) # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = 1 - (self.A)**2 # TODO
        
        return dAdZ


class ReLU:
    
    def forward(self, Z):

        self.A = (Z > 0) * Z # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = (self.A > 0) * 1 # TODO
        
        return dAdZ
        
        
