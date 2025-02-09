import numpy as np

import matplotlib.pyplot as plt
from src.maze_env import MazeEnv
from typing import List, Callable, Tuple
import matplotlib.animation as animation

# Define the states and possible actions
states = np.arange(1, 6)  # States 1 through 5
actions = ['left', 'stay', 'right']  # Available actions in each state


def transition(state, action):
    """
    Transition function that determines the next state based on the current state and action.

    Parameters:
    state (int): The current state.
    action (str): The action chosen.

    Returns:
    int: The next state.
    """
    if action == 'left':
        next_state = max(state -1, 1)
    elif action == 'right':
        next_state = min(state + 1, len(states))
    else:
        next_state = state

    return next_state


def reward(state, action):
    """
    Calculate the reward for a given state and action.

    Parameters:
    state (int): The current state.
    action (str): The action taken.

    Returns:
    int: The reward.
    """
    reward = 0

    if state == 4 and action == 'right':
        reward += 10
    else:
        reward -= 1
    return reward




def always_right_policy(state):
    """
    Policy that always returns 'right' for any given state.

    Parameters:
    state (int): The current state.

    Returns:
    str: The chosen action ('right').
    """
    if state <= 5:
        return 'right'

def my_policy(state):
    """
    This function implements a custom policy.

    Parameters:
    state (int): The current state of the system.

    Returns:
    str: The action chosen by the policy.
    """
    edge_state_actions = ['left', 'right']
    edge_state_probabilities = [0.5, 0.5]
    state_probabilities = [0.3, 0.3, 0.4]

    if state == 1 or state == 5:
        action = np.random.choice(edge_state_actions, p = edge_state_probabilities)
    else:
        action = np.random.choice(actions, p = state_probabilities)

    return action
        
def simulate_mdp(policy: Callable, initial_state=1, simulation_depth=20):
    """
    Simulates the Markov Decision Process (MDP) based on the given policy. 
    If we reach the terminal state, the simulation ends.
    Keeps track of the number of visits to each state, the cumulative reward, and the history of visited states.

    Parameters:
    - policy: A function that takes the current state as input and returns an action.
    - initial_state: The initial state of the MDP. Default is 1.
    - simulation_depth: The maximum number of steps to simulate. Default is 20.

    Returns:
    - state_visits: An array that tracks the number of visits to each state.
    - cumulative_reward: The cumulative reward obtained during the simulation.
    - visited_history: A list that tracks the history of visited states.
    - reward_history: A list that tracks the history of rewards obtained.
    """
    current_state = initial_state
    cumulative_reward = 0
    state_visits = np.zeros(len(states)) # Track the number of visits to each state
    visited_history = [current_state] # Track the history of visited states
    reward_history = [0] # Track the history of rewards
    
    for _ in range(simulation_depth):
        action = policy(current_state)
        next_state = transition(current_state, action)
        current_reward = reward(current_state, action)
        state_visits[next_state - 1] += 1
        cumulative_reward += current_reward
        visited_history.append(next_state)
        reward_history.append(current_reward)
        current_state = next_state

        if current_state == len(states):
            break

    return state_visits, cumulative_reward, visited_history, reward_history


def new_policy(state: List[int]) -> int:
    up_down = [0,1]
    left_right = [2,3]
    up_down_probabilities = [0.7, 0.3]
    left_right_probabilities = [0.3, 0.7]

    if state[1] < 3:
        next_action = np.random.choice(up_down, p = up_down_probabilities)
    else:
        next_action = np.random.choice(left_right, p = left_right_probabilities)

    return int(next_action)
        
def simulate_maze_env(env: MazeEnv, policy: Callable, num_steps=20):
    """
    Simulates the environment using the given policy for a specified number of steps.

    Parameters:
    - env: The environment to simulate.
    - policy: The policy to use for selecting actions (this is a function that takes a state as input and returns an action)
    - num_steps: The number of steps to simulate (default: 20).

    Returns:
    - path: The sequence of states visited during the simulation.
    - total_reward: The total reward accumulated during the simulation.
    """
    state = env.reset()
    total_reward = 0
    path = [state]

    for _ in range(num_steps):
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        path.append(next_state)
        state = next_state
        
        if done: 
            break

    return path, total_reward


def q_learning(env: MazeEnv, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1) -> np.ndarray:
    """
    Perform Q-learning to learn the optimal policy for the given environment.

    Args:
        env (MazeEnv): The environment to learn the policy for.
        episodes (int, optional): Number of episodes for training. Defaults to 500.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.99.
        epsilon (float, optional): Exploration rate. Defaults to 0.1.

    Returns:
        np.ndarray: The learned Q-table.
    """
    q_table = np.zeros((env.size, env.size, env.action_space.n))


    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            i, j = state

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[i, j, :]))

            next_state, reward, done, _ = env.step(action)
            next_i, next_j = next_state

            best_next_action_value = np.max(q_table[next_i, next_j, :])
            q_table[i, j, action] = q_table[i, j, action] + alpha * (reward + gamma * best_next_action_value - q_table[i, j, action])

            state = next_state

    return q_table


def simulate_maze_env_q_learning(
    env: MazeEnv, q_table: np.ndarray
) -> Tuple[List[Tuple[int, int]], bool]:
    """
    Simulate the maze environment using the Q-table to determine the actions to take.
    Also creates an animation of the agent moving through the environment.
    
    Args:
        env (MazeEnv): The maze environment instance.
        q_table (np.ndarray): The Q-table containing action values.

    Returns:
        Tuple[List[Tuple[int, int]], bool]: A tuple containing a list of states and a boolean indicating if the episode is done.
    """

    state = env.reset()
    done = False

    starting_frame = env.render(mode="rgb_array").T
    frames = [starting_frame]  # List to store frames for animation
    states = [state]  # List to store states

    while not done:
        i, j = state
        action = int(np.argmax(q_table[i, j, :]))
        state, _, done, _ = env.step(action)
        frames.append(
            env.render(mode="rgb_array").T
        )  # Render the environment as an RGB array
        states.append(state)

    def update_frame(i):
        ax.clear()
        ax.imshow(frames[i], cmap="viridis", origin="lower")
        ax.set_title(f"Step {i+1}")
        ax.grid("on")

    # Create animation from frames
    fig, ax = plt.subplots()
    anim = animation.FuncAnimation(fig, update_frame, frames=len(frames), interval=500)
    anim.save("mdp_q_learning.gif", writer="pillow")
    return states, done