import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

def make_env(is_slippery=True):
	env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=is_slippery)
	env.nS=env.observation_space.n
	env.nA=env.action_space.n
	return env

def plot_values(V):
	# reshape value function
	V_sq = np.reshape(V, (4,4))

	# plot the state-value function
	fig = plt.figure(figsize=(6, 6))
	ax = fig.add_subplot(111)
	im = ax.imshow(V_sq, cmap='cool')
	for (j,i),label in np.ndenumerate(V_sq):
	    ax.text(i, j, np.round(label, 5), ha='center', va='center', fontsize=14)
	plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
	plt.title('State-Value Function')
	plt.show()
