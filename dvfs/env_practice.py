from gymnasium import Env
# from gym.spaces import Box, Discrete
from gymnasium import spaces
import random
import numpy as np

import subprocess
from time import sleep
import os
import pickle

little_available_frequencies = [300000, 574000, 738000, 930000, 1098000, 1197000, 1328000, 1401000, 1598000, 1704000, 1803000]
mid_available_frequencies = [400000, 553000, 696000, 799000, 910000, 1024000, 1197000, 1328000, 1491000, 1663000, 1836000, 1999000, 2130000, 2253000]
big_available_frequencies = [500000, 851000, 984000, 1106000, 1277000, 1426000, 1582000, 1745000, 1826000, 2048000, 2188000, 2252000, 2401000, 2507000, 2630000, 2704000, 2802000]
gpu_available_frequencies = [151000, 202000, 251000, 302000, 351000, 400000, 471000, 510000, 572000, 701000, 762000, 848000]

little_len = 11
mid_len = 14
big_len = 17
gpu_len = 12

little_min_freq = 300000
mid_min_freq = 400000
big_min_freq = 500000
gpu_min_freq = 151000

little_max_freq = 1803000
mid_max_freq = 2253000
big_max_freq = 2802000 
gpu_max_freq = 848000

window = "asdf"

def get_window():
    msg = 'adb shell dumpsys SurfaceFlinger'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    result = result.split("\n")
    ans = "asdf"
    for i in range(len(result)):
        if "Current layers" in result[i]:
            print(result[i+4])
            ans = result[i+4]
            break
    ans = ans.split("[")
    ans = ans[1][:-1]

    window = ans
    return ans

# def get_fps():
    # import time

    # start = time.time()
    # msg = 'adb shell dumpsys gfxinfo org.chromium.webview_shell framestats'
    # msg = 'adb shell dumpsys SurfaceFlinger --latency ' + window
    # result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    # result = result.stdout.decode('utf-8')
    # result = result.split("\n")
    # end = time.time()
    # print(end - start)

def get_fps(window):
    msg = 'adb shell dumpsys SurfaceFlinger --latency ' + window
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    result = result.split("\n")

    startTime = int(result[-23].split("\t")[0])
    lastTime = int(result[-3].split("\t")[-1])  
    twentyFrameTime = (lastTime - startTime) / 1000000000
    fps = 20 / twentyFrameTime

    return fps

def action_to_freq(action):
    little_target, mid_target, big_target, gpu_target = action

    little_target = int(np.round(little_target/100*(little_len-1)))

    mid_target = int(np.round(mid_target/100*(mid_len-1)))

    big_target = int(np.round(big_target/100*(big_len-1)))

    gpu_target = int(np.round(gpu_target/100*(gpu_len-1)))


    little_min, little_max = little_available_frequencies[max(0, little_target)], little_available_frequencies[min(little_len-1, little_target)]
    mid_min, mid_max = mid_available_frequencies[max(0, mid_target)], mid_available_frequencies[min(mid_len-1, mid_target)]
    big_min, big_max = big_available_frequencies[max(0, big_target)], big_available_frequencies[min(big_len-1, big_target)]
    gpu_min, gpu_max = gpu_available_frequencies[max(0, gpu_target)], gpu_available_frequencies[min(gpu_len-1, gpu_target)]

    return little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max

def get_energy():
    msg = 'adb shell cat /sys/bus/iio/devices/iio:device0/energy_value'
    msg2 = 'adb shell cat /sys/bus/iio/devices/iio:device1/energy_value'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    result2 = subprocess.run(msg2.split(), stdout=subprocess.PIPE)

    result = result.stdout.decode('utf-8')
    result = result.split("\n")
    t1 = int(result[0][2:])
    big = int(result[4].split()[1])
    mid = int(result[5].split()[1])
    little = int(result[6].split()[1])
    
    result2 = result2.stdout.decode('utf-8')
    result2 = result2.split("\n")
    t2 = int(result2[0][2:])
    gpu = int(result2[7].split()[1])
    
    return t1, t2, little, mid, big, gpu

def get_temperatures():
    msg = 'adb shell cat /sys/devices/virtual/thermal/thermal_zone9/temp'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    big = int(result.stdout.decode('utf-8'))
    msg = 'adb shell cat /sys/devices/virtual/thermal/thermal_zone10/temp'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    mid = int(result.stdout.decode('utf-8'))
    msg = 'adb shell cat /sys/devices/virtual/thermal/thermal_zone11/temp'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    little = int(result.stdout.decode('utf-8'))
    msg = 'adb shell cat /sys/devices/virtual/thermal/thermal_zone12/temp'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    gpu = int(result.stdout.decode('utf-8'))
    msg = 'adb shell cat /sys/devices/virtual/thermal/thermal_zone23/temp'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    battery = int(result.stdout.decode('utf-8'))
    
    return little/1000, mid/1000, big/1000, gpu/1000, battery/1000

def set_frequency(little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max):
    """ little """
    msg = 'adb shell "echo '+str(little_min)+' > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    msg = 'adb shell "echo '+str(little_max)+' > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    """ mid """
    msg = 'adb shell "echo '+str(mid_min)+' > /sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    msg = 'adb shell "echo '+str(mid_max)+' > /sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    """ big """
    msg = 'adb shell "echo '+str(big_min)+' > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    msg = 'adb shell "echo '+str(big_max)+' > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    """ gpu """
    msg = 'adb shell "echo '+str(gpu_min)+' > /sys/class/misc/mali0/device/scaling_min_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    msg = 'adb shell "echo '+str(gpu_max)+' > /sys/class/misc/mali0/device/scaling_max_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()

def unset_frequency():
    """ little """
    msg = 'adb shell "echo '+str(little_min_freq)+' > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    msg = 'adb shell "echo '+str(little_max_freq)+' > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    """ mid """
    msg = 'adb shell "echo '+str(mid_min_freq)+' > /sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    msg = 'adb shell "echo '+str(mid_max_freq)+' > /sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    """ big """
    msg = 'adb shell "echo '+str(big_min_freq)+' > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    msg = 'adb shell "echo '+str(big_max_freq)+' > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    """ gpu """
    msg = 'adb shell "echo '+str(gpu_min_freq)+' > /sys/class/misc/mali0/device/scaling_min_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    msg = 'adb shell "echo '+str(gpu_max_freq)+' > /sys/class/misc/mali0/device/scaling_max_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()


def set_root(): 
    msg = 'adb root'
    subprocess.run(msg.split(), stdout=subprocess.PIPE)

def set_brightness(level):
    msg = 'adb shell settings put system screen_brightness '+str(level)
    subprocess.run(msg.split(), stdout=subprocess.PIPE)


class DVFStrain(Env):
    def __init__(self):
        # dog runs from 0 to 50, returns from 50 to 0


        # big temp, mid temp, little temp, gpu temp, battery temp
        # cpu utilization / memory
        # battery level
        # 
        self.observation_space = spaces.Box(low=0, high=100, shape=(5, ), dtype=np.float64)
        
        # amount of distance travelled 
        # self.action_space = spaces.Box(low=150999, high=2802001, shape=(8,), dtype=np.float64)
        self.action_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float64)
        
        # current state 
        self.state = np.array(get_temperatures())
        
        # no. of rounds
        self.rounds = 0
        
        # reward collected
        self.collected_reward = 0
    
        self.window = get_window()

        # adb root
        set_root()

        set_brightness(158)
    
    def step(self, action):


        # energy before
        t1a, t2a, littlea, mida, biga, gpua = get_energy()

        # set dvfs
        little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max = action_to_freq(action)
        set_frequency(little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max)

        # wait 1s
        sleep(0.5)

        fps = get_fps(self.window)

        # energy after
        t1b, t2b, littleb, midb, bigb, gpub = get_energy()

        # reward - energy
        little = (littleb - littlea)/(t1b-t1a)
        mid = (midb - mida)/(t1b-t1a)
        big = (bigb - biga)/(t1b-t1a)
        gpu = (gpub - gpua)/(t2b-t2a)

        reward = fps + 1000/(little+mid+big+gpu)

        # observation update
        obs = np.array(get_temperatures())
        info = {"little": little_min, "mid": mid_min, "big": big_min, "gpu": gpu_min}


        ac = [little_min, mid_min, big_min, gpu_min]
        # done = False
        # info = {}
        # rw = 0
        # self.rounds -= 1
        
        # obs = self.state + action
        
        # if obs < 50:
        #     self.collected_reward += -1
        #     rw = -1
        # elif obs > 50 and obs < 100:
        #     self.collected_reward += 0
        #     rw = 0
        # else:
        #     self.collected_reward += 1
        #     rw = 1
            
        # if self.rounds == 0:
        #     done = True

        self.collected_reward += reward

        self.rounds += 1
        
        self.render(ac, reward)
            
        return obs, reward, True, False, info
    
    def reset(self, seed, options):
        super().reset(seed=seed)
        self.state = np.array(get_temperatures())
        self.rounds = 0
        self.collected_reward = 0
        return self.state, {"a":1}
    
    def render(self, action, rw):
        print(f"Round : {self.rounds}\nDistance Travelled : {np.round(action[0]/1000000, 3), np.round(action[1]/1000000, 3), np.round(action[2]/1000000, 3), np.round(action[3]/1000000, 3)}\nReward Received: {rw}")
        print(f"Total Reward : {self.collected_reward}")
        print("=============================================================================")