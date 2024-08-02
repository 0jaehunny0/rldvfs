from utils import *
from time import sleep
from torch.utils.tensorboard import SummaryWriter
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--total_timesteps", type=int, default = 1001,
                    help="total timesteps of the experiments")
parser.add_argument("--experiment", type=int, default = 1,
                    help="the type of experiment")
parser.add_argument("--temperature", type=int, default = 20,
                    help="the ouside temperature")
parser.add_argument("--initSleep", type=int, default = 600,
                    help="initial sleep time")
parser.add_argument("--loadModel", type=str, default = "no",
                    help="initial sleep time")
parser.add_argument("--timeOut", type=int, default = 60*30,
                    help="end time")
args = parser.parse_args()

print(args)

total_timesteps = args.total_timesteps
experiment = args.experiment
temperature = args.temperature
initSleep = args.initSleep


# adb root
set_root()

turn_off_screen()

turn_on_screen()

sleep(initSleep)

turn_on_screen()

set_brightness(158)

window = get_window()

unset_frequency()

# turn_off_usb_charging()


unset_rate_limit_us()

msg = 'adb shell "echo '+"50000"+' > /sys/devices/system/cpu/cpufreq/policy0/sched_pixel/down_rate_limit_us"'
subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
msg = 'adb shell "echo '+"200000"+' > /sys/devices/system/cpu/cpufreq/policy4/sched_pixel/down_rate_limit_us"'
subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
msg = 'adb shell "echo '+"200000"+' > /sys/devices/system/cpu/cpufreq/policy6/sched_pixel/down_rate_limit_us"'
subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
msg = 'adb shell "echo '+"5000"+' > /sys/devices/system/cpu/cpufreq/policy0/sched_pixel/up_rate_limit_us"'
subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
msg = 'adb shell "echo '+"5000"+' > /sys/devices/system/cpu/cpufreq/policy4/sched_pixel/up_rate_limit_us"'
subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
msg = 'adb shell "echo '+"5000"+' > /sys/devices/system/cpu/cpufreq/policy6/sched_pixel/up_rate_limit_us"'
subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()

msg = 'adb shell "echo '+"200"+' > /sys/class/misc/mali0/device/dvfs_period"'
subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()     

run_name = "default3__" + str(int(time.time()))+"__exp"+str(experiment)+"__temp"+str(temperature)
writer = SummaryWriter(f"runs/{run_name}")

little = []
mid = []
big = []
gpu = []
ppwLi = []
fpsLi = []
powerLi = []
tempLi = []

start_time = time.time()

for i in range(total_timesteps):

    if time.time() - start_time > args.timeOut:
        break


    # energy before
    t1a, t2a, littlea, mida, biga, gpua = get_energy()

    sleep(1)

    fps = get_fps(window)

    freqs = np.array(get_frequency())

    # energy after
    t1b, t2b, littleb, midb, bigb, gpub = get_energy()

    little.append(freqs[0])
    mid.append(freqs[1])
    big.append(freqs[2])
    gpu.append(freqs[3])
    fpsLi.append(fps)
    tempLi.append(list(get_temperatures()))
    
    # reward - energy
    little_p = (littleb - littlea)/(t1b-t1a)
    mid_p = (midb - mida)/(t1b-t1a)
    big_p = (bigb - biga)/(t1b-t1a)
    gpu_p = (gpub - gpua)/(t2b-t2a)

    ppw = fps/(little_p + mid_p + big_p + gpu_p)

    ppwLi.append(ppw)
    powerLi.append(little_p + mid_p + big_p + gpu_p)

    if i % 10 == 0 and i != 0:
        print(i, end = " ")
        writer.add_scalar("freq/little", np.array(little)[-10:].mean(), i)
        writer.add_scalar("freq/mid", np.array(mid)[-10:].mean(), i)
        writer.add_scalar("freq/big", np.array(big)[-10:].mean(), i)
        writer.add_scalar("freq/gpu", np.array(gpu)[-10:].mean(), i)
        writer.add_scalar("perf/ppw", np.array(ppwLi)[-10:].mean(), i)
        writer.add_scalar("perf/power", np.array(powerLi)[-10:].mean(), i)
        writer.add_scalar("perf/fps", np.array(fpsLi)[-10:].mean(), i)
        writer.add_scalar("temp/little", np.array(tempLi)[-10:, 0].mean(), i)
        writer.add_scalar("temp/mid", np.array(tempLi)[-10:, 1].mean(), i)
        writer.add_scalar("temp/big", np.array(tempLi)[-10:, 2].mean(), i)
        writer.add_scalar("temp/gpu", np.array(tempLi)[-10:, 3].mean(), i)
        writer.add_scalar("temp/qi", np.array(tempLi)[-10:, 4].mean(), i)
        writer.add_scalar("temp/battery", np.array(tempLi)[-10:, 5].mean(), i)

        little_c, mid_c, big_c, gpu_c = get_cooling_state()
        writer.add_scalar("cstate/little", little_c, i)
        writer.add_scalar("cstate/mid", mid_c, i)
        writer.add_scalar("cstate/big", big_c, i)
        writer.add_scalar("cstate/gpu", gpu_c, i)


turn_on_usb_charging()
unset_rate_limit_us()
turn_off_screen()
unset_frequency()