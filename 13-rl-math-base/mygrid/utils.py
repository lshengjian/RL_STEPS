import numpy as np
from gymnasium.utils import seeding
from random import choice
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

default_rand_generator=seeding.np_random(1234)[0]

    

def update_policy(pi_s:np.ndarray,max_q_action:int,eps):
    nA=len(pi_s)
    pi_s[:]=eps/nA
    pi_s[max_q_action]=1-(pi_s[0]*nA-1)
    
def categorical_sample(prob_n, np_random: np.random.Generator=default_rand_generator):
    """Sample from categorical distribution where each row specifies class probabilities."""
    prob_n = np.asarray(prob_n)
    prob_n/=prob_n.sum()  #0~1
    csprob_n = np.cumsum(prob_n)
    #print(csprob_n)
    #assert csprob_n[-1]==1.0
    cond=csprob_n > np_random.random()
    idx=np.argmax(cond)
    #print(idx,cond)
    return idx

def greedy_select(logics, np_random: np.random.Generator=default_rand_generator):
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

def plot_steps_and_rewards(rewards_df, steps_df,savefn='./data/demo.png'):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(
        data=rewards_df, x="Episodes", y="cum_rewards", hue="map_size", ax=ax[0]
    )
    ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=steps_df, x="Episodes", y="Steps", hue="map_size", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")

    for axi in ax:
        axi.legend(title="map size")
    fig.tight_layout()
    img_title = "frozenlake_steps_and_rewards.png"
    fig.savefig(savefn, bbox_inches="tight")
    plt.show()
# def plot_values(V):
# 	# reshape value function
# 	V_sq = np.reshape(V, (4,4))

# 	# plot the state-value function
# 	fig = plt.figure(figsize=(6, 6))
# 	ax = fig.add_subplot(111)
# 	im = ax.imshow(V_sq, cmap='cool')
#     for (j,i),label in np.ndenumerate(V_sq):
#         ax.text(i, j, np.round(label, 5), ha='center', va='center', fontsize=14)
#     plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
# 	plt.title('State-Value Function')
# 	plt.show()
        
if __name__ == "__main__":
   data=[1.0,2.0,3.0,3.0]
   d=greedy_select(data)
   print(d)
   d=greedy_select(data)
   print(d)
   res_all = pd.DataFrame()
   st_all = pd.DataFrame()