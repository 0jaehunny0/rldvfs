from utils import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--temp", type=int, default = 40,
                    help="target init battery temperature")
args = parser.parse_args()

print(args)

wait_temp(args.temp)