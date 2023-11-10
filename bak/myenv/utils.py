import numpy as np

def max_dict(d):
  # returns the argmax (key) and max (value) from a dictionary
  max_key = None
  max_val = float('-inf')
  for k, v in d.items():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val

def random_action(a, acts,eps=0.1,):
  # epsilon-soft to ensure all states are visited
  p = np.random.random()
  if p> eps:
    return np.random.choice(acts)
  else:
    return a
    