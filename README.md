Step by step learning reinforcement learning

# install Gymnasium on Windows 11

## Classic Control
``` python
pip install gymnasium[classic-control]
```
## Box2d
download [swigwin](http://prdownloads.sourceforge.net/swig/swigwin-4.1.1.zip) and add its directory to system PATH
``` python
pip install gymnasium[classic-box2d]
```
## MuJoCo
- download [mojoco 2.1.5](https://github.com/deepmind/mujoco/releases?page=2)
- unzip to yuor home directory .mujoco and rename it to mujoco210
- add C:/Users/[your account]/.mujoco/mujoco210/bin to system PATH
``` python
pip install gymnasium[classic-mujoco]
```

# Acknowledgements

thank [MorvanZhou](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)

thank [seungeunrho](https://github.com/seungeunrho/minimalRL)


