import csv
import matplotlib.pyplot as plt
import numpy as np


episodes = []
rewards = []
steps = []

# =====================
# READ CSV
# =====================
with open("training_log8.csv", "r") as f:

    reader = csv.reader(f)

    for row in reader:

        episode = int(row[0])

        episodes.append(episode)
        rewards.append(float(row[1]))
        steps.append(int(row[2]))


# =====================
# MOVING AVERAGE
# =====================
WINDOW = 20

reward_ma = []

for i in range(len(rewards)):

    if i < WINDOW - 1:

        reward_ma.append(None)

    else:

        avg = sum(
            rewards[i - WINDOW + 1:i + 1]
        ) / WINDOW

        reward_ma.append(avg)


# =====================
# STATISTICS
# =====================
reward_mean = np.mean(rewards)
reward_std = np.std(rewards)

step_mean = np.mean(steps)
step_std = np.std(steps)

print("Reward Statistics")
print("-----------------")
print(f"Mean: {reward_mean:.2f}")
print(f"Std : {reward_std:.2f}")

print()

print("Step Statistics")
print("-----------------")
print(f"Mean: {step_mean:.2f}")
print(f"Std : {step_std:.2f}")


# =====================
# PLOT
# =====================
plt.figure(figsize=(12, 6))

plt.plot(
    episodes,
    rewards,
    alpha=0.3,
    label="Raw Reward"
)

plt.plot(
    episodes,
    reward_ma,
    linewidth=3,
    label=f"{WINDOW}-Episode Moving Average"
)

plt.axhline(
    reward_mean,
    linestyle="--",
    linewidth=2,
    label=f"Mean Reward ({reward_mean:.2f})"
)

plt.xlabel("Episode")
plt.ylabel("Reward")

plt.title(
    f"Training Progress\n"
    f"Reward Mean={reward_mean:.2f}, Std={reward_std:.2f} | "
    f"Step Mean={step_mean:.2f}, Std={step_std:.2f}"
)

plt.legend()
plt.grid(True)

plt.show()