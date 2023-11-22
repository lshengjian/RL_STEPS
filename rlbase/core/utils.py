import numpy as np
from random import choice
import numpy as np
   
    
def categorical_sample(prob_n,np_random: np.random.Generator): 
    """Sample from categorical distribution where each row specifies class probabilities."""
    prob_n = np.asarray(prob_n)
    prob_n/=prob_n.sum()  #0~1
    csprob_n = np.cumsum(prob_n)
    #print(csprob_n)
    #assert csprob_n[-1]==1.0
    cond=csprob_n > np.random.random()
    idx=np.argmax(cond)
    #print(idx,cond)
    return idx

def greedy_select(logics, np_random: np.random.Generator):
    """Sample from categorical distribution where each row specifies class probabilities."""
    if len(logics)<1 :
        return 0
    logics = np.asarray(logics)
    
    total=logics.sum()
    if total==0 :
        return 0 
    logics/=logics.sum()  #0~1
    best_a = np.argwhere(logics==np.max(logics)).flatten()
    return choice(best_a)

def greedy_select(logics, np_random: np.random.Generator,explore=False):
    """Sample from categorical distribution where each row specifies class probabilities."""
    if len(logics)<1 :
        return 0
    logics = np.asarray(logics)
    
    total=logics.sum()
    if total==0 :
        return 0 
    logics/=logics.sum()  #0~1
    fn=np.min if explore else np.max
    best_a = np.argwhere(logics==fn(logics)).flatten()
    return choice(best_a)