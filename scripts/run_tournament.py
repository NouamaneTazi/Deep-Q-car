"""
To test a trained agent on an environment from the carl/ directory run

python3 -m scripts.run_tournament
"""

import argparse

from src.circuit import Circuit
from src.tournament import TournamentEnvironment, Tournament


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_laps', type=int, default=3)
    args = parser.parse_args()

    circuit = Circuit(
        [(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (6, 0)], width=0.3)
    env = TournamentEnvironment(circuit, render=True, laps=args.num_laps)
    tournament = Tournament(env, 10000)
    tournament.run()
    print('\n'.join(map(str, tournament.scores)))
