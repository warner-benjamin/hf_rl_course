import argparse
import sys

from dqn import atari_dqn
from dqn import atari_dqn_ema

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, help="what script to run: atari_dqn, atari_dqn_ema")
    args, _ = parser.parse_known_args()

    if args.run == 'atari_dqn':
        atari_dqn.train(sys.argv, True)
    elif args.run == 'atari_dqn_ema':
        atari_dqn_ema.train(sys.argv, True)