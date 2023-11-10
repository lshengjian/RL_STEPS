from agents import *
from envs import make_env
from utils import *
from config import *


import numpy as np
import pickle


def main():
    print({section: dict(config[section]) for section in config.sections()})
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']
    env = make_env(render_mode='human')


    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2
    print(input_size,output_size)

    

    is_render = True
    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)

    use_cuda = False
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step  / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])

    sticky_action = False
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')



    agent = RNDAgent(
        input_size,
        output_size,
        1,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net
    )

    print('Loading Pre-trained model....')
    if use_cuda:
        agent.model.load_state_dict(torch.load(model_path))
        agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
        agent.rnd.target.load_state_dict(torch.load(target_path))
    else:
        agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
        agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
    print('End load...')


    #states = np.zeros([num_worker, 4, 84, 84])
    state,info=env.reset()
    state=np.array(state)
    state=state.reshape([1, 4, 84, 84])
    print(state.shape)

    steps = 0
    end = False
    intrinsic_reward_list = []
    while not end:
        steps += 1
        actions, value_ext, value_int, policy = agent.get_action(state)

        next_state, reward, done,trunc,info=env.step(actions[0])
        next_state=np.array(next_state)
        next_state=next_state.reshape([1, 4, 84, 84])
        
        state = next_state[:, :, :, :]
        # intrinsic_reward = agent.compute_intrinsic_reward(state[0][-1].reshape([1, 1, 84, 84]))
        # print(intrinsic_reward)
 

        if done:
            end=True
            steps = 0



if __name__ == '__main__':
    main()
