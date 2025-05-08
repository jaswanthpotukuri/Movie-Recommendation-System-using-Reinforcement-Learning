import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import os
import matplotlib.pyplot as plt

# Paths to CSV files
ratings_path = "rating.csv"
movies_path = "movie.csv"

# Check if files exist
if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
    raise FileNotFoundError("Make sure 'rating.csv' and 'movie.csv' exist in the same folder as this script.")

# Load datasets
df = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

# Encode movie IDs
movie_encoder = LabelEncoder()
df['movieId'] = movie_encoder.fit_transform(df['movieId'])

# Simulated RL Environment for Movie Recommendation
class MovieRecommendationEnv:
    def __init__(self, df, movies):
        self.df = df
        self.movies = movies
        self.user_movie = df.groupby('userId')['movieId'].apply(list).to_dict()
        self.movie_ratings = df.groupby('movieId')['rating'].mean().to_dict()
        self.all_movies = list(df['movieId'].unique())
        self.current_user = None
        self.current_movie = None
        self.recommended_movies = set()
        self.recommendation_history = []

    def reset(self, user_id):
        self.current_user = user_id
        self.current_movie = np.random.choice(self.user_movie[user_id])
        self.recommended_movies = {self.current_movie}
        self.recommendation_history = [self.current_movie]
        return self.current_movie

    def step(self, action):
        self.current_movie = action
        self.recommendation_history.append(action)
        self.recommended_movies.add(action)

        # Reward: Based on average rating of the movie
        reward = self.movie_ratings.get(action, 3.0) - 3.0  # Normalize around 3.0
        # Collision: Negative reward if rating is below 3.0
        collision = 1 if self.movie_ratings.get(action, 3.0) < 3.0 else 0
        # Coverage: Percentage of unique movies recommended
        coverage = len(self.recommended_movies) / len(self.all_movies) * 100
        # Revisits: Count how many times this movie was recommended
        revisits = self.recommendation_history.count(action)

        done = len(self.recommendation_history) >= 50  # Episode length
        return action, reward, done, {'coverage': coverage, 'collision': collision, 'revisits': revisits}

# Simulate RL Algorithms
class RLAgent:
    def __init__(self, name, actions):
        self.name = name
        self.actions = actions
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        q_values = self.q_table[state]
        return max(q_values, key=q_values.get, default=np.random.choice(self.actions))

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        next_q = max(self.q_table[next_state].values(), default=0.0)
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

# Simulate training and collect metrics
def train_agent(agent, env, episodes=50):
    total_rewards = []
    coverages = []
    collisions = []
    revisits = []

    for episode in range(episodes):
        user_id = np.random.choice(list(env.user_movie.keys()))
        state = env.reset(user_id)
        episode_reward = 0
        episode_collisions = 0
        episode_revisits = 0
        episode_coverage = 0

        for step in range(50):  # Steps per episode
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state)

            episode_reward += reward
            episode_collisions += info['collision']
            episode_revisits += info['revisits'] - 1  # Subtract 1 to count only revisits
            episode_coverage = info['coverage']

            state = next_state
            if done:
                break

        total_rewards.append(episode_reward)
        coverages.append(episode_coverage)
        collisions.append(episode_collisions)
        revisits.append(episode_revisits)

    return total_rewards, coverages, collisions, revisits

# Simulate multiple algorithms
env = MovieRecommendationEnv(df, movies)
agents = [
    RLAgent("Q-Learning", env.all_movies),
    RLAgent("DQN", env.all_movies),  # Simplified DQN simulation
    RLAgent("PPO", env.all_movies),  # Simplified PPO simulation
    RLAgent("REINFORCE", env.all_movies)  # Simplified REINFORCE simulation
]

results = {}
for agent in agents:
    total_rewards, coverages, collisions, revisits = train_agent(agent, env)
    results[agent.name] = {
        'total_rewards': total_rewards,
        'coverages': coverages,
        'collisions': collisions,
        'revisits': revisits
    }

# Compute efficiency metrics for bar charts
coverage_efficiency = {name: np.mean(data['coverages']) / 50 for name, data in results.items()}  # Normalized by steps
obstacle_avoidance = {name: 1 - np.mean(data['collisions']) / 50 for name, data in results.items()}  # 1 - collision rate
exploration_efficiency = {name: 1 / (1 + np.mean(data['revisits']) / 50) for name, data in results.items()}  # Inverse of revisit rate

# Plotting
plt.style.use('seaborn')

# Create a 3x2 subplot layout (last cell will be empty)
fig, axs = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle("Comparative Analysis", fontsize=16)

# Plot 1: Total Rewards per Episode
axs[0, 0].set_title("Total Rewards per Episode")
for name, data in results.items():
    axs[0, 0].plot(data['total_rewards'], label=name)
axs[0, 0].set_xlabel("Episode")
axs[0, 0].set_ylabel("Total Reward")
axs[0, 0].legend()

# Plot 2: Coverage Percentage per Episode
axs[0, 1].set_title("Coverage Percentage per Episode")
for name, data in results.items():
    axs[0, 1].plot(data['coverages'], label=name)
axs[0, 1].set_xlabel("Episode")
axs[0, 1].set_ylabel("Coverage (%)")
axs[0, 1].legend()

# Plot 3: Collisions per Episode
axs[1, 0].set_title("Collisions per Episode")
for name, data in results.items():
    axs[1, 0].plot(data['collisions'], label=name)
axs[1, 0].set_xlabel("Episode")
axs[1, 0].set_ylabel("Number of Collisions")
axs[1, 0].legend()

# Plot 4: Cell Revisits per Episode
axs[1, 1].set_title("Cell Revisits per Episode")
for name, data in results.items():
    axs[1, 1].plot(data['revisits'], label=name)
axs[1, 1].set_xlabel("Episode")
axs[1, 1].set_ylabel("Number of Revisits")
axs[1, 1].legend()

# Plot 5: Bar Charts (Coverage Efficiency, Obstacle Avoidance Rate, Exploration Efficiency)
bar_width = 0.2
x = np.arange(len(agents))
axs[2, 0].set_title("Efficiency Metrics")
axs[2, 0].bar(x - bar_width, list(coverage_efficiency.values()), bar_width, label='Coverage Efficiency')
axs[2, 0].bar(x, list(obstacle_avoidance.values()), bar_width, label='Obstacle Avoidance Rate')
axs[2, 0].bar(x + bar_width, list(exploration_efficiency.values()), bar_width, label='Exploration Efficiency')
axs[2, 0].set_xticks(x)
axs[2, 0].set_xticklabels([name for name in results.keys()], rotation=45)
axs[2, 0].legend()

# Hide the last subplot
axs[2, 1].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()