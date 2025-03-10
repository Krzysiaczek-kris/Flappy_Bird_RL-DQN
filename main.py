# main.py
import gymnasium as gym
from flappy_bird_env import FlappyBirdEnv, MAX_SCORE
from dqn_agent import DQNAgent
import numpy as np
import time
from datetime import timedelta
import csv
import torch

torch.backends.cudnn.benchmark = True # Improves performance for fixed input size

env = FlappyBirdEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
episodes = 50000
batch_size = 128
log_batch_size = 10
start = time.time()

consecutive_max_scores = 0

with open('training_log.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['episode', 'reward', 'score', 'steps', 'time'])

log_buffer = []

for e in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    terminated = False
    steps_in_episode = 0

    while not terminated:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        steps_in_episode += 1
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state[0], action, reward, next_state[0], terminated)
        state = next_state
        total_reward += reward

    score = env.score

    if score == MAX_SCORE:
        consecutive_max_scores += 1
        print(f"Episode {e+1}, Max Score Achieved! Consecutive Max Scores: {consecutive_max_scores}")
        if consecutive_max_scores >= 50:
            print("Agent has achieved maximum score for 50 consecutive episodes. Stopping training.")
            break
    else:
        consecutive_max_scores = 0
        agent.replay(batch_size)
        agent.update_target_network()

    log_buffer.append([
        e + 1,
        total_reward,
        score,
        steps_in_episode,
        str(timedelta(seconds=time.time() - start))
    ])

    elapsed = time.time() - start
    print(f"Episode {e+1}, Reward: {total_reward}, Epsilon: {agent.epsilon:.6f}, Score: {score}, Time: {str(timedelta(seconds=elapsed))[:-4]}")

    if (e + 1) % log_batch_size == 0:
        with open('training_log.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(log_buffer)
        log_buffer.clear()

    # if (e + 1) % 100 == 0:
    #     torch.save(agent.model.state_dict(), f"model.pth")

if log_buffer:
    with open('training_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(log_buffer)

torch.save(agent.model.state_dict(), f"model.pth")
print("Training complete.")
