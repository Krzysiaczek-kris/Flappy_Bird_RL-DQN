import torch
import numpy as np
import time
from datetime import timedelta
import csv
from flappy_bird_env import FlappyBirdEnv
from dqn_agent import DQNAgent
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

log_lock = threading.Lock()

def test_model(n_episodes, thread_id):
    env = FlappyBirdEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    agent.epsilon = 0
    agent.model.load_state_dict(torch.load("model.pth", weights_only=True))

    start = time.time()

    for episode in range(n_episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        terminated = False
        total_reward = 0

        while not terminated and env.isopen:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = np.reshape(next_state, [1, state_size])
            total_reward += reward

            # env.render()

        score = env.score
        elapsed = time.time() - start
        log_entry = [thread_id, episode+1, total_reward, score, str(timedelta(seconds=elapsed))[:-4]]
        print(f"Thread {thread_id}, Episode {episode+1}, Reward: {total_reward}, Score: {score}, Time: {log_entry[3]}")

        with log_lock:
            with open('testing_log.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(log_entry)
                file.flush() 

    env.close()


def main(n_episodes, k):
    episodes_per_thread = n_episodes // k
    remainder = n_episodes % k

    with open('testing_log.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['thread', 'episode', 'reward', 'score', 'time'])
        file.flush() 

    futures = []
    with ThreadPoolExecutor(max_workers=k) as executor:
        start_episode = 1
        for thread_id in range(k):
            thread_episodes = episodes_per_thread + (1 if thread_id < remainder else 0)
            futures.append(
                executor.submit(test_model, thread_episodes, thread_id)
            )
            start_episode += thread_episodes

        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    n_episodes = 10
    k_threads = 2
    main(n_episodes, k_threads)
