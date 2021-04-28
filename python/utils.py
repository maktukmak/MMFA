import numpy as np
from numpy.linalg import inv

def softmax(x):
    tmp = np.append(x, np.zeros((1, x.shape[1])), axis = 0)
    e_x = np.log(np.sum(np.exp(tmp - np.amax(tmp,axis=0,keepdims=True)), axis = 0, keepdims=True)) + np.amax(tmp,axis=0,keepdims=True)
    
    out = tmp - e_x
    prob = np.exp(out)
    
    return prob

def logsumexp(x):
    tmp = np.append(x, np.zeros((1, x.shape[1])), axis = 0)
    lse = np.log(np.sum(np.exp(tmp - np.amax(tmp,axis=0,keepdims=True)), axis = 0, keepdims=True)) + np.amax(tmp,axis=0,keepdims=True)
    
    return lse

def logdet(x):
    
    return np.log(np.linalg.det(x))

