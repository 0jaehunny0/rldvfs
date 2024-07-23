from utils2 import *
from time import sleep
from torch.utils.tensorboard import SummaryWriter
import time

# adb root
set_root()

# turn_on_screen()

set_brightness(158)

window = get_window()

unset_frequency()

unset_rate_limit_us()

run_name = "temperatureTest__" + str(int(time.time()))
writer = SummaryWriter(f"runs/{run_name}")

little = []
mid = []
big = []
gpu = []
ppwLi = []

temps = []

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

for i in range(300):

    temps.append(list(get_all_temperatures()))
    sleep(0.5)

turn_off_screen()

for i in range(2000):

    sleep(0.5)



a=1
print(2)

x = np.array(temps)
import pandas as pd

x = pd.DataFrame(x)


x.columns = ["big", "mid", "little", "gpu", "qi", "battery", "disp", "gnss", "neutral", "TPU", "ISP", "quiet", "usb1", "usb2"]
# x.columns = ["qi", "disp", "gnss", "battery", "quiet", "neutral", "tpu", "isp", "usb", "rf1", "rf2"]

import matplotlib.pyplot as plt
x.plot()




import pickle
with open('temp exp.pkl', 'wb') as f:
	pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)