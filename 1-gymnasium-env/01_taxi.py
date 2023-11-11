import gymnasium as gym

NUM_EPISODES = 10

def evaluate_agent(env:gym.Env, num_trials):
    total_epochs, total_penalties = 0, 0
    acts=env.action_space
    print("Running episodes...")
    for _ in range(num_trials):
        state,info = env.reset()
        epochs, num_penalties, reward = 0, 0, 0
        while reward != 20:
            next_action = acts.sample(info['action_mask'])
            state, reward, _, _,info = env.step(next_action)
            if reward == -10:
                num_penalties += 1
            epochs += 1
        total_penalties += num_penalties
        total_epochs += epochs

    average_time = total_epochs / float(num_trials)
    average_penalties = total_penalties / float(num_trials)
    print("Evaluation results after {} trials".format(num_trials))
    print("Average time steps taken: {}".format(average_time))
    print("Average number of penalties incurred: {}".format(average_penalties))


def main(num_episodes):
    env = gym.make("Taxi-v3",render_mode='human')
    evaluate_agent(env, num_episodes)


if __name__ == "__main__":
    main(NUM_EPISODES)
