import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from collections import OrderedDict
import numpy as np

from bandits import BernoulliBandit, DriftingBandit 
from agents import EpsilonGreedyAgent, UCBAgent, ThompsonSamplingAgent

np.random.seed(42)

def get_regret(env, agents, n_steps=5000, n_trials=50):
    scores = OrderedDict({
        agent.name: [0.0 for step in range(n_steps)] for agent in agents
    })

    for trial in range(n_trials):
        env.reset()

        for a in agents:
            a.init_actions(env.action_count)

        for i in range(n_steps):
            optimal_reward = env.optimal_reward()

            for agent in agents:
                action = agent.get_action()
                reward = env.pull(action)
                agent.update(action, reward)
                scores[agent.name][i] += optimal_reward - env.action_value(action)

    for agent in agents:
        scores[agent.name] = np.cumsum(scores[agent.name]) / n_trials

    return scores

def plot_regret(agents, scores, save_path=None):
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    
    plt.figure(figsize=(10, 8))

    for agent in agents:
        plt.plot(scores[agent.name], linestyle=next(linecycler))

    plt.legend([agent.name for agent in agents])

    plt.ylabel("Cumulative Regret")
    plt.xlabel("Steps")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

if __name__ == "__main__":
    agents = [
        EpsilonGreedyAgent(),
        UCBAgent(),
        ThompsonSamplingAgent()
    ]

    regret = get_regret(BernoulliBandit(), agents, n_steps=10000, n_trials=10)
    plot_regret(agents, regret, save_path="regret_plot.png")