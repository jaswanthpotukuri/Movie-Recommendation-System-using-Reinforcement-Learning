🎬 Movie Recommendation System using MDP, MAB, and DQN
This project explores reinforcement learning techniques to build an intelligent Movie Recommendation System. It integrates three core approaches—Markov Decision Processes (MDP), Multi-Armed Bandit (MAB), and Deep Q-Networks (DQN)—to dynamically recommend movies based on user preferences and interactions.

🔍 Overview
Goal: Recommend personalized movies to users by modeling and adapting to their changing preferences.

Dataset: MovieLens 20M — includes user ratings, movie genres, and timestamps.

🧠 Techniques Used
1. Markov Decision Process (MDP)
Models the recommendation environment as states (user preferences) and actions (movie recommendations).

Uses transition probabilities and rewards (user feedback) to suggest the best next movie.

2. Multi-Armed Bandit (MAB)
Implements the ε-greedy algorithm to balance exploration (trying new genres) and exploitation (recommending known favorites).

Quickly adapts to changing user preferences with minimal prior data.

3. Deep Q-Network (DQN)
Applies deep reinforcement learning to handle large state and action spaces.

Trains a neural network to approximate Q-values and make optimal recommendations.

Learns from user interaction data to improve over time.

💡 Key Features
Real-time learning and updating based on simulated user feedback.

Handles cold-start scenarios and dynamic preference shifts.

Modular structure to switch between MDP, MAB, and DQN methods.
