from utils import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--temp", type=float, default = 37.5,
                    help="target init battery temperature")
args = parser.parse_args()

print(args)

set_root()

turn_on_usb_charging()
unset_rate_limit_us()
turn_off_screen()
unset_frequency()

wait_temp(args.temp)