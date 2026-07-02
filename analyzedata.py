import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

episodes = []
rewards = []
steps = []

# Read CSV
with open("training_log6.csv", "r") as f:
    reader = csv.reader(f)

    for row in reader:
        episode = int(row[0])

        # Only use the first 200 episodes
       # if episode >= 200:
       #     break

        episodes.append(episode) 
        rewards.append(float(row[1]))
        steps.append(int(row[2]))

# ----- Moving Average -----
WINDOW = 20
reward_ma = []

for i in range(len(rewards)):
    if i < WINDOW - 1:
        reward_ma.append(None)  # Not enough data yet
    else:
        avg = sum(rewards[i-WINDOW+1:i+1]) / WINDOW
        reward_ma.append(avg)

# ----- Linear Regression -----
X = np.array(episodes).reshape(-1, 1)
y = np.array(rewards)

model = LinearRegression()
model.fit(X, y)

trend = model.predict(X)

print(f"Slope: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R²: {model.score(X, y):.4f}")

# ----- Plot -----
plt.figure(figsize=(12, 6))

plt.plot(episodes, rewards, alpha=0.3, label="Raw Reward")
plt.plot(episodes, reward_ma, linewidth=3, label=f"{WINDOW}-Episode Moving Average")
plt.plot(episodes, trend, "--", linewidth=2, label="Linear Regression")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Progress")
plt.legend()
plt.grid(True)

plt.show()