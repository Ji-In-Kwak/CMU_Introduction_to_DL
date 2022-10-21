import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):
        
        self.alpha     = alpha
        self.eps       = 1e-8
        
        self.Z         = None
        self.NZ        = None
        self.BZ        = None

        self.BW        = np.ones((1, num_features))
        self.Bb        = np.zeros((1, num_features))
        self.dLdBW     = np.zeros((1, num_features))
        self.dLdBb     = np.zeros((1, num_features))
        
        self.M         = np.zeros((1, num_features))
        self.V         = np.ones((1, num_features))
        
        # inference parameters
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """
        
        if eval:
            ones      = np.ones((Z.shape[0], 1))
            NZ        = (Z - np.dot(ones, self.running_M)) / np.dot(ones, np.sqrt(self.running_V + self.eps)) 
            BZ        = np.dot(ones, self.BW) * NZ + np.dot(ones, self.Bb)
            # TODO
            return BZ
            
        self.Z         = Z
        self.N         = self.Z.shape[0] # TODO
        
        self.M         = np.mean(Z, axis=0).reshape(1, -1) # TODO
        self.V         = np.var(Z, axis=0).reshape(1, -1) # TODO
        self.ones      = np.ones((self.N, 1))
        self.NZ        = (self.Z - np.dot(self.ones, self.M)) / np.dot(self.ones, np.sqrt(self.V + self.eps)) # TODO
        self.BZ        = np.dot(self.ones, self.BW) * self.NZ + np.dot(self.ones, self.Bb) # TODO
        
        self.running_M = self.alpha * self.running_M + (1-self.alpha) * self.M # TODO
        self.running_V = self.alpha * self.running_V + (1-self.alpha) * self.V # TODO
        
        return self.BZ

    def backward(self, dLdBZ):
        
        self.dLdBW  = np.sum(dLdBZ * self.NZ, axis=0) # TODO
        self.dLdBb  = np.sum(dLdBZ, axis=0) # TODO
        
#         print('mu and sigma = \n', self.M, self.V)
        dLdNZ       = dLdBZ * self.BW # TODO
#         print('dLdNZ = \n', dLdNZ)
        dLdV        = (-1/2) * (dLdNZ * (self.Z - self.M) * (self.V + self.eps)**(-3/2)) # TODO
        dLdV        = np.sum(dLdV, axis=0)
#         print('dLdV = \n', np.sum(dLdV, axis=0))
        dLdM        = -dLdNZ * (self.V + self.eps)**(-1/2) - (2/self.N) * dLdV * (self.Z - self.M) # TODO
        dLdM        = np.sum(dLdM, axis=0)
#         print('dLdM = \n', np.sum(dLdM, axis=0))

        dLdZ        = dLdNZ / np.sqrt(self.V + self.eps) + dLdV * (2*(self.Z - self.M)/self.N) + dLdM / self.N # TODO
        
        return  dLdZ