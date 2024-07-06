from gymnasium import Env
# from gym.spaces import Box, Discrete
from gymnasium import spaces
import random
import numpy as np

import subprocess
from time import sleep

from utils import *

from collections import deque

fpsDeque = deque()
temp_ths = 45

def action_to_freq(action):

    little_min, little_max = little_available_frequencies[action[0]], little_available_frequencies[action[0]]
    mid_min, mid_max = mid_available_frequencies[action[1]], mid_available_frequencies[action[1]]
    big_min, big_max = big_available_frequencies[action[2]], big_available_frequencies[action[2]]
    gpu_min, gpu_max = gpu_available_frequencies[action[3]], gpu_available_frequencies[action[3]]

    return little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max

def cal_reward(fps, little, mid, big, gpu, temps):

    reward = 0
    l_t, m_t, b_t, g_t, qi_t, bat_t = temps
    
    min_fps = min(fpsDeque)
    max_fps = max(fpsDeque)

    target_fps = (min_fps + max_fps) / 2

    alpha, beta, gamma = 0, 0, 0

    beta = fps / (little + mid + big + gpu)

    # if fps < target_fps:
    #     alpha = -1 * beta * (target_fps - fps) / (target_fps - min_fps)
    
    # if qi_t > temp_ths or disp_t > temp_ths:
    #     gamma = -1 * beta * ( max(qi_t, disp_t) - temp_ths) / temp_ths

    reward = alpha + beta + gamma


    return reward

class DVFStrain(Env):
    def __init__(self):

        # fps, temp(6), freq(4), power(4), cluster_util (4), 
        self.observation_space = spaces.Box(low=0, high=100, shape=(19, ), dtype=np.float64)
        
        self.action_space = spaces.MultiDiscrete([11, 14, 17, 12])
                
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

        states = np.concatenate([[fps], [little_u, mid_u, big_u], gpu_util, [little, mid, big, gpu], temps, freqs/30000]).astype(np.float32)

        self.state = states

        # states = np.concatenate([gpu_util,cpu_util,gpu_freq,cpu_freq,gpu_thremal,cpu_freq,power]).astype(np.float32)

    def step(self, action):

        # energy before
        t1a, t2a, littlea, mida, biga, gpua = get_energy()
        a = get_core_util()

        # set dvfs
        little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max = action_to_freq(action)
        set_frequency(little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max)
        freqs = np.array([little_min, mid_min, big_min, gpu_min])

        # wait 0.5s
        sleep(0.1)

        # current state 
        temps = np.array(get_temperatures())
        # freqs = np.array(get_frequency())

        fps = get_fps(self.window)

        if len(fpsDeque) >= 100:
            fpsDeque.pop()
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

        states = np.concatenate([[fps], [little_u, mid_u, big_u], gpu_util, [little, mid, big, gpu], temps, freqs/30000]).astype(np.float32)

        self.state = states
        obs = states

        reward = cal_reward(fps, little, mid, big, gpu, temps)

        ppw = fps/(little*100 + mid*100 + big*100 + gpu*100)

        # observation update
        info = {"little": little_min, "mid": mid_min, "big": big_min, "gpu": gpu_min, "fps":fps, "power":little + mid + big + gpu, "reward":reward, "ppw":ppw, "temp":temps}

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