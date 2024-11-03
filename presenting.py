# presenting.py
import torch
import numpy as np
from flappy_bird_env import FlappyBirdEnv
from dqn_agent import DQNAgent

def run_game():
    env = FlappyBirdEnv()

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    agent.epsilon = 0

    agent.model.load_state_dict(torch.load("model.pth"))

    while env.isopen:
        try:
            state, _ = env.reset()
            state = np.reshape(state, [1, state_size])
            terminated = False

            while not terminated and env.isopen:
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                state = np.reshape(next_state, [1, state_size])

                env.render()
        except Exception as e:
            print(f"Game crashed: {e}. Restarting...")
    env.close()

if __name__ == "__main__":
    run_game()