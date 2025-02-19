from tqdm import tqdm
from .blackjack import BlackjackEnv, Hand, Card

ACTIONS = ['hit', 'stick']

def policy_evaluation(env, V, policy, episodes=500000, gamma=1.0):
    """
    Monte Carlo policy evaluation:
    - Generate episodes using the current policy
    - Update state value function as an average return
    """
    # Initialize returns_sum and returns_count
    returns_sums = {}
    returns_count = {}
    for _ in tqdm(range(episodes), desc="Policy evaluation"):
        # Generate one episode
        episode = []
        state = env.reset()
        while True:
            action = policy.get(state)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done: 
                break

        for i, (state, action, reward) in enumerate(episode):
            G = 0
            discount = 1.0
            for j in range(i,len(episode)):
                G += discount * episode[j][2]
                discount *= gamma
                
                if state not in returns_sums:
                    returns_sums[state] = 0.0
                    returns_count[state] = 0
                
                returns_sums[state] += G
                returns_count[state] += 1

    # Update V(s) as the average return
    for state in returns_sums:
        V[state] = returns_sums[state] / returns_count[state]
    return V
