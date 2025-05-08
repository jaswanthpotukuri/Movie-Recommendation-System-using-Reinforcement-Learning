import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import os

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

# user-movie interactions
user_movie = df.groupby('userId')['movieId'].apply(list).to_dict()

# Define MDP model
class MovieRecommenderMDP:
    def __init__(self, df, gamma=0.9):
        self.df = df
        self.gamma = gamma
        self.states = defaultdict(list)
        self.actions = list(df['movieId'].unique())
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.rewards = defaultdict(lambda: defaultdict(float))
        self._build_mdp()

    def _build_mdp(self):
        for user_id, group in self.df.groupby('userId'):
            movies_seq = group.sort_values('timestamp')['movieId'].tolist()
            for i in range(len(movies_seq) - 1):
                s = movies_seq[i]
                a = movies_seq[i + 1]
                self.states[user_id].append(s)
                self.transitions[s][a] += 1
                self.rewards[s][a] += 1

        # Normalize transitions and rewards
        for s in self.transitions:
            total = sum(self.transitions[s].values())
            for a in self.transitions[s]:
                self.transitions[s][a] /= total
                self.rewards[s][a] /= total

    def value_iteration(self, iterations=50):
        V = defaultdict(float)
        for _ in range(iterations):
            new_V = defaultdict(float)
            for s in self.transitions:
                new_V[s] = max(
                    (self.transitions[s][a] * (self.rewards[s][a] + self.gamma * V[a])
                     for a in self.transitions[s]), default=0.0
                )
            V = new_V
        self.V = V

    def recommend(self, current_movie, top_k=5):
        if current_movie not in self.transitions:
            return []
        q_values = {}
        for a in self.transitions[current_movie]:
            q_values[a] = self.rewards[current_movie][a] + self.gamma * self.V[a]
        top_recommendations = sorted(q_values.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(movie_encoder.inverse_transform([m])[0], score) for m, score in top_recommendations]

# Initialize and run value iteration
mdp = MovieRecommenderMDP(df)
mdp.value_iteration()

# Sample recommendation output
sample_movie = next(iter(mdp.transitions))
sample_movie_original_id = movie_encoder.inverse_transform([sample_movie])[0]
print(f"\n📽️ Recommendations based on: {sample_movie_original_id}")

recommendations = mdp.recommend(sample_movie)
if not recommendations:
    print("❌ No recommendations found.")
else:
    for title_id, score in recommendations:
        title_row = movies[movies['movieId'] == title_id]
        title = title_row['title'].values[0] if not title_row.empty else f"Movie ID {title_id}"
        print(f"✅ {title} (Score: {score:.2f})")

# Evaluation function: Hit Rate@k
def evaluate_model(mdp, top_k=5):
    hits = 0
    total = 0

    for user_id, group in df.groupby('userId'):
        movies_seq = group.sort_values('timestamp')['movieId'].tolist()

        for i in range(len(movies_seq) - 1):
            current_movie = movies_seq[i]
            actual_next_movie = movies_seq[i + 1]

            if current_movie not in mdp.transitions:
                continue

            recommended = mdp.recommend(current_movie, top_k=top_k)
            recommended_ids = [movie_encoder.transform([m])[0] for m, _ in recommended]

            if actual_next_movie in recommended_ids:
                hits += 1
            total += 1

    accuracy = hits / total if total > 0 else 0
    print(f"\n🎯 Hit Rate@{top_k}: {accuracy:.4f} ({hits} hits out of {total} transitions)")
    return accuracy

# Run evaluation
evaluate_model(mdp, top_k=5)
