from utils2 import *
from time import sleep
from torch.utils.tensorboard import SummaryWriter
import time
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--little", type=int, default = 1197000,
                    help="little frequency")
args = parser.parse_args()


def get_all_temperatures():

    msg = 'adb shell cat /dev/thermal/tz-by-name/BIG/temp /dev/thermal/tz-by-name/MID/temp /dev/thermal/tz-by-name/LITTLE/temp /dev/thermal/tz-by-name/G3D/temp /dev/thermal/tz-by-name/qi_therm/temp /dev/thermal/tz-by-name/battery/temp'
    msg += ' /dev/thermal/tz-by-name/disp_therm/temp /dev/thermal/tz-by-name/gnss_tcxo_therm/temp /dev/thermal/tz-by-name/neutral_therm/temp /dev/thermal/tz-by-name/TPU/temp /dev/thermal/tz-by-name/ISP/temp /dev/thermal/tz-by-name/quiet_therm/temp /dev/thermal/tz-by-name/usb_pwr_therm/temp /dev/thermal/tz-by-name/usb_pwr_therm2/temp'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    result = result.split("\n")
    big = int(result[0])
    mid = int(result[1])
    little = int(result[2])
    gpu = int(result[3])
    qi = int(result[4])
    battery = int(result[5])
    disp = int(result[6])
    gnss = int(result[7])
    neutral = int(result[8])
    TPU = int(result[9])
    ISP = int(result[10])
    quiet = int(result[11])
    usb1 = int(result[12])
    usb2 = int(result[13])

    return big, mid, little, gpu, qi, battery, disp, gnss, neutral, TPU, ISP, quiet, usb1, usb2

def offCores():

    msg = 'adb shell "echo 0 > /sys/devices/system/cpu/cpu4/online'
    msg += ' && echo 0 > /sys/devices/system/cpu/cpu5/online'
    msg += ' && echo 0 > /sys/devices/system/cpu/cpu6/online'
    msg += ' && echo 0 > /sys/devices/system/cpu/cpu7/online'
    msg += '"'

    result = subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()


def onCores():

    msg = 'adb shell "echo 1 > /sys/devices/system/cpu/cpu4/online'
    msg += ' && echo 1 > /sys/devices/system/cpu/cpu5/online'
    msg += ' && echo 1 > /sys/devices/system/cpu/cpu6/online'
    msg += ' && echo 1 > /sys/devices/system/cpu/cpu7/online'
    msg += '"'

    result = subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    

def setLittle(little_min, little_max, gpu_min, gpu_max):

    msg = 'adb shell "echo '+str(little_min)+' > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq'
    msg += ' && echo '+str(little_max)+' > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq'

    msg += ' && echo '+str(gpu_min)+' > /sys/class/misc/mali0/device/scaling_min_freq'
    msg += ' &&  echo '+str(gpu_max)+' > /sys/class/misc/mali0/device/scaling_max_freq'
    msg += '"'

    result = subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()

# adb root
set_root()
onCores()
unset_frequency()
unset_rate_limit_us()

run_name = "motivation__temp__" + str(int(time.time()))
writer = SummaryWriter(f"runs/{run_name}")

little = []
mid = []
big = []
gpu = []
ppwLi = []
temps = []




turn_on_screen()
set_brightness(158)
turn_off_screen()
unset_frequency()
unset_rate_limit_us()


offCores()
setLittle(args.little, args.little, 151000, 151000)
# setLittle(738000, 738000, 151000, 151000)
# setLittle(1197000, 1197000, 151000, 151000)
# setLittle(1598000, 1598000, 151000, 151000)

turn_off_screen()
# for i in range(15000):
for i in range(6000):
    res = list(get_all_temperatures())
    temps.append(res)
    if i % 5 == 0:
        writer.add_scalar("temp/big", res[0], i)
        writer.add_scalar("temp/mid", res[1], i)
        writer.add_scalar("temp/little", res[2], i)
        writer.add_scalar("temp/gpu", res[3], i)
        writer.add_scalar("temp/qi", res[4], i)
        writer.add_scalar("temp/battery", res[5], i)
        writer.add_scalar("temp/disp", res[6], i)
    sleep(0.2)

turn_on_screen()
# for i in range(15000, 15000+3000):
for i in range(6000, 6000 + 3000):
    res = list(get_all_temperatures())
    temps.append(res)
    if i % 5 == 0:
        writer.add_scalar("temp/big", res[0], i)
        writer.add_scalar("temp/mid", res[1], i)
        writer.add_scalar("temp/little", res[2], i)
        writer.add_scalar("temp/gpu", res[3], i)
        writer.add_scalar("temp/qi", res[4], i)
        writer.add_scalar("temp/battery", res[5], i)
        writer.add_scalar("temp/disp", res[6], i)
    sleep(0.2)


onCores() 
turn_off_screen()
unset_rate_limit_us()
unset_frequency()
writer.close()

x = np.array(temps)

x = pd.DataFrame(x)


x.columns = ["big", "mid", "little", "gpu", "qi", "battery", "disp", "gnss", "neutral", "TPU", "ISP", "quiet", "usb1", "usb2"]

# x.plot()

with open('motivation_temp'+str(args.little)+'.pkl', 'wb') as f:
	pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)     

with open('motivation_temp'+str(args.little)+'.pkl', 'rb') as f:
	x = pickle.load(f)
