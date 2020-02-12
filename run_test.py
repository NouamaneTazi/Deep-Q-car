"""
To test a trained agent on an environment from the carl/ directory run

python3 -m scripts.run_test
"""

import argparse
import os.path

from src.agent import DQLAgent
from src.circuit import Circuit
from src.environment import Environment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    agent = DQLAgent(gamma=args.gamma, max_steps=args.max_steps)
    circuit = Circuit(
        [(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (6, 0)], width=0.3)

    env = Environment(circuit, render=True)
    if agent.load(args.model):
        name = os.path.basename(args.model)
        agent.run_once(env, train=False, greedy=True, name=name[:-3])
        print("{:.2f} laps in {} steps".format(
            circuit.laps + circuit.progression, args.max_steps))
