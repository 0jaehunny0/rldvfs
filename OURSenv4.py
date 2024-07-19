from gymnasium import Env
# from gym.spaces import Box, Discrete
from gymnasium import spaces
import random
import numpy as np

import subprocess
from time import sleep

from utils import *

from collections import deque

# def get_temperatures():

#     msg = 'adb shell "echo 2048000 > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq && echo 2048000 > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq && echo 738000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq && echo 738000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq && echo 1491000 > /sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq && echo 1491000 > /sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq && echo 510000 > /sys/class/misc/mali0/device/scaling_min_freq &&  echo 510000 > /sys/class/misc/mali0/device/scaling_max_freq'
#     msg += ' && cat /sys/devices/virtual/thermal/thermal_zone9/temp /sys/devices/virtual/thermal/thermal_zone10/temp /sys/devices/virtual/thermal/thermal_zone11/temp /sys/devices/virtual/thermal/thermal_zone12/temp /sys/devices/virtual/thermal/thermal_zone17/temp /sys/devices/virtual/thermal/thermal_zone23/temp'
#     msg += '"'
#     result = subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
#     result = result.decode('utf-8')
#     # result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
#     # result = result.stdout.decode('utf-8')
#     # result = result.split("\n")


#     msg = 'adb shell cat /sys/devices/virtual/thermal/thermal_zone9/temp /sys/devices/virtual/thermal/thermal_zone10/temp /sys/devices/virtual/thermal/thermal_zone11/temp /sys/devices/virtual/thermal/thermal_zone12/temp /sys/devices/virtual/thermal/thermal_zone17/temp /sys/devices/virtual/thermal/thermal_zone23/temp'
#     result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
#     result = result.stdout.decode('utf-8')
#     result = result.split("\n")
#     big = int(result[0])
#     mid = int(result[1])
#     little = int(result[2])
#     gpu = int(result[3])
#     qi = int(result[4])
#     battery = int(result[5])

#     return little/1000, mid/1000, big/1000, gpu/1000, qi/1000, battery/1000

# def get_adb():
#     msg = 'adb shell cat /dev/thermal/cdev-by-name/thermal-cpufreq-0/cur_state /dev/thermal/cdev-by-name/thermal-cpufreq-1/cur_state /dev/thermal/cdev-by-name/thermal-cpufreq-2/cur_state /dev/thermal/cdev-by-name/thermal-gpufreq-0/cur_state'
#     msg = 'adb shell cat /sys/devices/virtual/thermal/thermal_zone9/temp /sys/devices/virtual/thermal/thermal_zone10/temp /sys/devices/virtual/thermal/thermal_zone11/temp /sys/devices/virtual/thermal/thermal_zone12/temp /sys/devices/virtual/thermal/thermal_zone17/temp /sys/devices/virtual/thermal/thermal_zone23/temp'
#     msg = msg + ' /sys/bus/iio/devices/iio:device0/energy_value /sys/bus/iio/devices/iio:device1/energy_value'
#     result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
#     result = result.stdout.decode('utf-8')
#     result = result.split("\n")
#     little_cdev = int(result[0])
#     mid_cdev = int(result[1])
#     big_cdev = int(result[2])
#     gpu_cdev = int(result[3])

#     # """ big """
#     # msg = 'adb shell "echo '+str(big_min)+' > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq && ' + "echo '+str(big_max)+' > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq"
#     # subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
#     # msg = 'adb shell "echo '+str(big_max)+' > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq"'
#     # subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()

#     # msg = 'adb shell "echo '+str(big_min)+' > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq'
#     # msg += ' && echo '+str(big_max)+' > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq'

#     # msg += ' && echo '+str(little_min)+' > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq'
#     # msg += ' && echo '+str(little_max)+' > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq'

#     # msg += ' && echo '+str(mid_min)+' > /sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq'
#     # msg += ' && echo '+str(mid_max)+' > /sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq'


#     # msg += ' && echo '+str(gpu_min)+' > /sys/class/misc/mali0/device/scaling_min_freq'
#     # msg += ' &&  echo '+str(gpu_max)+' > /sys/class/misc/mali0/device/scaling_max_freq'

#     # msg += '"'

#     # subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    

def get_cooling_state():
    msg = 'adb shell cat /dev/thermal/cdev-by-name/thermal-cpufreq-0/cur_state /dev/thermal/cdev-by-name/thermal-cpufreq-1/cur_state /dev/thermal/cdev-by-name/thermal-cpufreq-2/cur_state /dev/thermal/cdev-by-name/thermal-gpufreq-0/cur_state'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    result = result.split("\n")
    little = int(result[0])
    mid = int(result[1])
    big = int(result[2])
    gpu = int(result[3])
    return little, mid, big, gpu

fpsDeque = deque()
temp_ths = 45

def action_to_freq(action, c_states):
    
    li = int(min(action[0], little_len - c_states[0]))
    mi = int(min(action[1], mid_len - c_states[1]))
    bi = int(min(action[2], big_len - c_states[2]))
    gi = int(min(action[3], gpu_len - c_states[3]))

    little_min, little_max = little_available_frequencies[li], little_available_frequencies[li]
    mid_min, mid_max = mid_available_frequencies[mi], mid_available_frequencies[mi]
    big_min, big_max = big_available_frequencies[bi], big_available_frequencies[bi]
    gpu_min, gpu_max = gpu_available_frequencies[gi], gpu_available_frequencies[gi]

    return little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max

def cal_reward(fps, little, mid, big, gpu, freqs, c_states):

    reward = 0

    reward = fps / (little + mid + big + gpu)

    target_fps = sum(list(fpsDeque)[-150:]) / min(150, len(fpsDeque))

    if fps < target_fps:
        reward *= fps/target_fps    

    return reward

class DVFStrain(Env):
    def __init__(self):

        # fps, temp(6), freq(4), power(4), cluster_util (4), cooling_state (4), prev_time
        self.observation_space = spaces.Box(low=0, high=100, shape=(24, ), dtype=np.float64)
        
        self.action_space = spaces.MultiDiscrete([11, 14, 17, 12, 50])
                
        # no. of rounds
        self.rounds = 0
        
        # reward collected
        self.collected_reward = 0

        # adb root
        set_root()
        
        turn_on_screen()

        set_brightness(158)

        self.window = get_window()

        turn_off_usb_charging()

        sleep(600)

        set_rate_limit_us(10000000, 20000)

        # energy before
        t1a, t2a, littlea, mida, biga, gpua = get_energy()
        a = get_core_util()

        # wait 0.5s
        sleep(0.1)

        # current state 
        temps = np.array(get_temperatures())
        freqs = np.array(get_frequency())

        fps = get_fps(self.window)

        fpsDeque.appendleft(fps)

        # energy after
        t1b, t2b, littleb, midb, bigb, gpub = get_energy()

        # reward - energy
        little = (littleb - littlea)/(t1b-t1a)/100
        mid = (midb - mida)/(t1b-t1a)/100
        big = (bigb - biga)/(t1b-t1a)/100
        gpu = (gpub - gpua)/(t2b-t2a)/100

        b = get_core_util()
        gpu_util = [get_gpu_util()[0] * 100]
        
        cpu_util = np.array(list(cal_core_util(b,a)))

        little_u = cpu_util[0:4].mean()*100
        mid_u = cpu_util[4:6].mean()*100
        big_u = cpu_util[6:8].mean()*100

        c_states = list(get_cooling_state())

        states = np.concatenate([[fps], [little_u, mid_u, big_u], gpu_util, [little, mid, big, gpu], temps, freqs/30000, c_states, [0]]).astype(np.float32)

        self.state = states

        # states = np.concatenate([gpu_util,cpu_util,gpu_freq,cpu_freq,gpu_thremal,cpu_freq,power]).astype(np.float32)

    def step(self, action):

        c_states = [self.state[-5], self.state[-4], self.state[-3], self.state[-2]]
        # set dvfs
        little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max = action_to_freq(action, c_states)
        freqs = np.array([little_min, mid_min, big_min, gpu_min])
        t1a, t2a, littlea, mida, biga, gpua, a = set_frequency_and_get_energy(little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max)

        # wait 0.5s
        sleep(0.1 * (action[-1]+1))

        c_states, temps, fps, t1b, t2b, littleb, midb, bigb, gpub, b, gpu_util = (get_states(self.window))

        if len(fpsDeque) >= 100:
            fpsDeque.pop()
        fpsDeque.appendleft(fps)

        # reward - energy
        little = (littleb - littlea)/(t1b-t1a)/100
        mid = (midb - mida)/(t1b-t1a)/100
        big = (bigb - biga)/(t1b-t1a)/100
        gpu = (gpub - gpua)/(t2b-t2a)/100
        
        cpu_util = np.array(list(cal_core_util(b,a)))

        little_u = cpu_util[0:4].mean()*100
        mid_u = cpu_util[4:6].mean()*100
        big_u = cpu_util[6:8].mean()*100

        states = np.concatenate([[fps], [little_u, mid_u, big_u], gpu_util, [little, mid, big, gpu], temps, freqs/30000, c_states, [action[-1]]]).astype(np.float32)

        self.state = states
        obs = states

        reward = cal_reward(fps, little, mid, big, gpu, action, c_states)

        ppw = fps/(little*100 + mid*100 + big*100 + gpu*100)

        times = 0.1 * (action[-1]+1)

        # observation update
        info = {"little": little_min, "mid": mid_min, "big": big_min, "gpu": gpu_min, "fps":fps, "power":little + mid + big + gpu, "reward":reward, "ppw":ppw, "temp":temps, "time":times}

        ac = [little_min, mid_min, big_min, gpu_min]

        self.collected_reward += reward

        self.rounds += 1
        
        self.render(ac, reward)
            
        return obs, reward, True, False, info
    
    def reset(self, seed, options):
        super().reset(seed=seed)
        # self.state = 
        # self.rounds = 0
        self.collected_reward = 0
        return self.state, {"a":1}
    
    def render(self, action, rw):
        print(f"Round : {self.rounds}\nDistance Travelled : {np.round(action[0]/1000000, 3), np.round(action[1]/1000000, 3), np.round(action[2]/1000000, 3), np.round(action[3]/1000000, 3)}\nReward Received: {rw}")
        print(f"Total Reward : {self.collected_reward}")
        print("=============================================================================")