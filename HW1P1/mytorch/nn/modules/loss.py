import numpy as np

class MSELoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        se     = (A - Y)**2 # TODO
        sse    = np.matmul(np.matmul(np.ones(N), se), np.ones(C)) # TODO
        mse    = sse/(N*C)
        
        return mse
    
    def backward(self):
    
        dLdA = (self.A - self.Y)
        
        return dLdA

class CrossEntropyLoss:
    
    def forward(self, A, Y):
    
        self.A   = A
        self.Y   = Y
        N        = A.shape[0]
        C        = A.shape[1]
        Ones_C   = np.ones((C, 1), dtype="f")
        Ones_N   = np.ones((N, 1), dtype="f")
        
        self.softmax     = np.exp(A) / np.matmul(np.matmul(np.exp(A), Ones_C), np.transpose(Ones_C)) # TODO
        crossentropy     = - self.Y * np.log(self.softmax) # TODO
        sum_crossentropy = np.matmul(np.matmul(np.transpose(Ones_N), crossentropy), Ones_C) # TODO
        L = sum_crossentropy / N
        
        return L
    
    def backward(self):
        
#         N        = self.A.shape[0]
#         C        = self.A.shape[1]
#         Ones_C  = np.ones((N, C), dtype="f")
#         dsoftmax = self.softmax(Ones_C - self.softmax)
#         dLdA = - self.Y * (Ones_C - self.softmax) # TODO
        dLdA = self.softmax - self.Y
        
        return dLdA
