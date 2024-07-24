from gymnasium import Env
# from gym.spaces import Box, Discrete
from gymnasium import spaces
import random
import numpy as np

import subprocess
from time import sleep

from utils2 import *

from collections import deque


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

    little_max = little_available_frequencies[li]
    mid_max = mid_available_frequencies[mi]
    big_max = big_available_frequencies[bi]
    gpu_max = gpu_available_frequencies[gi]

    sleepTime = 0.1 * (int(action[4]) + 1)
    up = 500 + 500 * int(action[5])
    down = 500 + 500 * int(action[6])
    gpu = 20 + 10 * int(action[7])

    return little_max, mid_max, big_max, gpu_max, sleepTime, up, down, gpu

def cal_reward(fps, little, mid, big, gpu, action, c_states):

    temp = little_len + mid_len + big_len + gpu_len

    reward = 0



    li = max(0, action[0] - (little_len - c_states[0]))
    mi = max(0, action[1] - (mid_len - c_states[1]))
    bi = max(0, action[2] - (big_len - c_states[2]))
    gi = max(0, action[3] - (gpu_len - c_states[3]))

    miss = li + mi + bi + gi
    miss = ((temp - miss) / temp) ** 0.5

    c_state_reward = ((temp - sum(c_states)) / temp) ** 0.5

    # reward = fps / (little + mid + big + gpu) * c_state_reward
    reward = fps / (little + mid + big + gpu) * miss * c_state_reward

    # target_fps = sum(list(fpsDeque)[-100:]) / min(100, len(fpsDeque))

    # if fps < target_fps:
    #     reward *= fps/target_fps    

    return reward

class DVFStrain(Env):
    def __init__(self, initSleep, experiment):

        self.exp = experiment

        # fps, temp(6), freq(4), power(4), cluster_util (4), cooling_state (4), prev_states (8)
        self.observation_space = spaces.Box(low=0, high=100, shape=(31, ), dtype=np.float64)
        
        # max_limit[little mid big gpu] repeat [up_rate down_rate (500 - 10000)] gpu_rate (20-100)
        self.action_space = spaces.MultiDiscrete([11, 14, 17, 12, 10, 20, 20, 9])
                
        # no. of rounds
        self.rounds = 0
        
        # reward collected
        self.collected_reward = 0

        # adb root
        set_root()
        
        turn_off_screen()

        turn_on_screen()
        
        sleep(initSleep)

        turn_on_screen()

        set_brightness(158)

        self.window = get_window()

        turn_off_usb_charging()


        set_rate_limit_us2(500, 5000, 20)

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

        states = np.concatenate([[fps], [little_u, mid_u, big_u], gpu_util, [little, mid, big, gpu], temps, freqs/30000, [0, 0, 0, 0, 0, 0, 9, 0], c_states]).astype(np.float32)

        self.state = states

        # states = np.concatenate([gpu_util,cpu_util,gpu_freq,cpu_freq,gpu_thremal,cpu_freq,power]).astype(np.float32)

    def step(self, action):

        c_states = [self.state[-5], self.state[-4], self.state[-3], self.state[-2]]
        # set limit freq and 
        little_max, mid_max, big_max, gpu_max, sleepTime, up, down, gpu_rate = action_to_freq(action, c_states)
        # freqs = np.array([little_max, mid_min, big_min, gpu_min])
        t1a, t2a, littlea, mida, biga, gpua, a = set_frequency_and_get_energy2(little_max, mid_max, big_max, gpu_max, up, down, gpu_rate)

        # wait 0.5s
        sleep(sleepTime)

        c_states, temps, fps, t1b, t2b, littleb, midb, bigb, gpub, b, gpu_util, freqs = (get_states2(self.window))

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

        states = np.concatenate([[fps], [little_u, mid_u, big_u], gpu_util, [little, mid, big, gpu], temps, freqs/30000, action, c_states]).astype(np.float32)

        self.state = states
        obs = states

        reward = cal_reward(fps, little, mid, big, gpu, action, c_states)

        ppw = fps/(little*100 + mid*100 + big*100 + gpu*100)

        

        # observation update
        info = {"little": freqs[0], "mid": freqs[1], "big": freqs[2], "gpu": freqs[3], "fps":fps, "power":little + mid + big + gpu, "reward":reward, "ppw":ppw, "temp":temps, "time":sleepTime, "uptime":up, "downtime":down, "gputime":gpu_rate}

        ac = freqs

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
        if self.rounds % 10 == 0:
            print(self.rounds, end = " ")
        # print(f"Round : {self.rounds}\nDistance Travelled : {np.round(action[0]/1000000, 3), np.round(action[1]/1000000, 3), np.round(action[2]/1000000, 3), np.round(action[3]/1000000, 3)}\nReward Received: {rw}")
        # print(f"Total Reward : {self.collected_reward}")
        # print("=============================================================================")