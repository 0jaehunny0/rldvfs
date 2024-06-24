from gymnasium import Env
# from gym.spaces import Box, Discrete
from gymnasium import spaces
import random
import numpy as np

import subprocess
from time import sleep
import os
import pickle
import math


little_available_frequencies = [300000, 574000, 738000, 930000, 1098000, 1197000, 1328000, 1401000, 1598000, 1704000, 1803000]
mid_available_frequencies = [400000, 553000, 696000, 799000, 910000, 1024000, 1197000, 1328000, 1491000, 1663000, 1836000, 1999000, 2130000, 2253000]
big_available_frequencies = [500000, 851000, 984000, 1106000, 1277000, 1426000, 1582000, 1745000, 1826000, 2048000, 2188000, 2252000, 2401000, 2507000, 2630000, 2704000, 2802000]
gpu_available_frequencies = [151000, 202000, 251000, 302000, 351000, 400000, 471000, 510000, 572000, 701000, 762000, 848000]


little_available_frequencies = np.array(little_available_frequencies)[[int(np.round(11*2/8))-1, int(np.round(11*5/8))-1, int(np.round(11*8/8))-1]]
mid_available_frequencies = np.array(mid_available_frequencies)[[int(np.round(14*2/8))-1, int(np.round(14*5/8))-1, int(np.round(14*8/8))-1]]
big_available_frequencies = np.array(big_available_frequencies)[[int(np.round(17*2/8))-1, int(np.round(17*5/8))-1, int(np.round(17*8/8))-1]]
gpu_available_frequencies = np.array(gpu_available_frequencies)[[int(np.round(12*1/4))-1, int(np.round(12*2/4))-1, int(np.round(12*3/4))-1]]

# int(np.round(11*2/8))-1, int(np.round(11*5/8))-1, int(np.round(11*8/8))-1
# int(np.round(14*2/8))-1, int(np.round(14*5/8))-1, int(np.round(14*8/8))-1
# int(np.round(17*2/8))-1, int(np.round(17*5/8))-1, int(np.round(17*8/8))-1
# int(np.round(12*1/4))-1, int(np.round(12*2/4))-1, int(np.round(12*3/4))-1

target_fps=60
target_temp=65
beta=2

clk_action_list = []
for i in range(3):
    for j in range(3):
        clk_action=(i,j)
        clk_action_list.append(clk_action)


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

def get_reward(fps, power, target_fps, c_t, g_t, c_t_prev, g_t_prev, beta):
	v1=0
	v2=0
	print('power={}'.format(power))
	u=max(1,fps/target_fps)

	if g_t<= target_temp:
		v2=0
	else:
		v2=2*(target_temp-g_t)
	if c_t_prev < target_temp:
		if c_t >= target_temp:
			v1=-2

	if fps>=target_fps:
		u=1
	else :
		u=math.exp(0.1*(fps-target_fps))
	return u+v1+v2+beta/power

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

    cpu_index = clk_action_list[action][0]
    gpu_index = clk_action_list[action][1]


    little_min, little_max = little_available_frequencies[cpu_index], little_available_frequencies[cpu_index]
    mid_min, mid_max = mid_available_frequencies[cpu_index], mid_available_frequencies[cpu_index]
    big_min, big_max = big_available_frequencies[cpu_index], big_available_frequencies[cpu_index]
    gpu_min, gpu_max = gpu_available_frequencies[gpu_index], gpu_available_frequencies[gpu_index]

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


def ternary (n):
    if n == 0:
        return '0000'
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    res = ''.join(reversed(nums))
    if len(res) == 1:
        return "000" + res
    elif len(res) == 2:
        return "00" + res
    elif len(res) == 3:
        return "0" + res
    return res
class DVFStrain(Env):
    def __init__(self):
 
    
        self.window = get_window()

        # adb root
        set_root()

        set_brightness(158)

        """
        cpu_freq 1/2/3 
        gpu freq_1/2/3 
        cpu power
        gpu power
        cpu temp
        gpu temp
        fps
        """
        self.observation_space = spaces.Box(low=0, high=100, shape=(7, ), dtype=np.float64)        

        self.action_space = spaces.Discrete(9)
        


        # energy before
        t1a, t2a, littlea, mida, biga, gpua = get_energy()

        # set dvfs
        little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max = action_to_freq(8)
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

        l_t, m_t, b_t, g_t, bat_t  = get_temperatures()

        self.state = np.array([3, 3, (little + mid + big), gpu, m_t, g_t, fps])
        
        self.c_t_prev = m_t
        self.g_t_prev = g_t

        # no. of rounds
        self.rounds = 0
        
        # reward collected
        self.collected_reward = 0
     
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


        l_t, m_t, b_t, g_t, bat_t  = get_temperatures()

        reward = get_reward(fps, little + mid + big + gpu, target_fps, m_t, g_t, self.c_t_prev, self.g_t_prev, beta)

        self.c_t_prev = m_t
        self.g_t_prev = g_t

        info = {"little": little_min, "mid": mid_min, "big": big_min, "gpu": gpu_min}

        ac = [little_min, mid_min, big_min, gpu_min]

        obs = np.array([3, 3, (little + mid + big), gpu, m_t, g_t, fps])

        self.collected_reward += reward

        self.rounds += 1
        
        self.render(ac, reward)
            
        return obs, reward, True, False, info
    
    def reset(self, seed, options):
        super().reset(seed=seed)


        # energy before
        t1a, t2a, littlea, mida, biga, gpua = get_energy()

        # set dvfs
        little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max = action_to_freq(8)
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

        l_t, m_t, b_t, g_t, bat_t  = get_temperatures()

        self.state = np.array([3, 3, (little + mid + big), gpu, m_t, g_t, fps])
        
        self.c_t_prev = m_t
        self.g_t_prev = g_t
        
        self.rounds = 0
        self.collected_reward = 0
        return self.state, {"a":1}
    
    def render(self, action, rw):
        print(f"Round : {self.rounds}\nDistance Travelled : {np.round(action[0]/1000000, 3), np.round(action[1]/1000000, 3), np.round(action[2]/1000000, 3), np.round(action[3]/1000000, 3)}\nReward Received: {rw}")
        print(f"Total Reward : {self.collected_reward}")
        print("=============================================================================")