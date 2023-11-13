import numpy as np
from gymnasium.utils import seeding

default_rand_generator=seeding.np_random(1234)[0]
def categorical_sample(prob_n, np_random: np.random.Generator=default_rand_generator):
    """Sample from categorical distribution where each row specifies class probabilities."""
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    #print(csprob_n)
    #assert csprob_n[-1]==1.0
    cond=csprob_n > np_random.random()
    idx=np.argmax(cond)
    #print(idx,cond)
    return idx

