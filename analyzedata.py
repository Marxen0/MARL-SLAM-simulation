import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

episodes = []
rewards = []
steps = []

EPISODE_START = 0
EPISODE_END = 500      # Set to None to read until the end

with open("Version 21 Training/training_log.csv", "r") as f:

    reader = csv.reader(f)

    first_row = next(reader)

    def process_row(row):
        if len(row) < 3:
            return

        episode = int(row[0])

        if episode < EPISODE_START:
            return

        if EPISODE_END is not None and episode > EPISODE_END:
            return

        episodes.append(episode)
        rewards.append(float(row[1]))
        steps.append(int(row[2]))

    try:
        process_row(first_row)

    except ValueError:
        # First row is a header
        pass

    for row in reader:
        process_row(row)
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