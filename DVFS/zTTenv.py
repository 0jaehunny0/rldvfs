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

from utils import *
import time

little_available_frequencies = np.array(little_available_frequencies)[[int(np.round(11*2/8))-1, int(np.round(11*5/8))-1, int(np.round(11*8/8))-1]]
mid_available_frequencies = np.array(mid_available_frequencies)[[int(np.round(14*2/8))-1, int(np.round(14*5/8))-1, int(np.round(14*8/8))-1]]
big_available_frequencies = np.array(big_available_frequencies)[[int(np.round(17*2/8))-1, int(np.round(17*5/8))-1, int(np.round(17*8/8))-1]]
gpu_available_frequencies = np.array(gpu_available_frequencies)[[int(np.round(12*1/4))-1, int(np.round(12*2/4))-1, int(np.round(12*3/4))-1]]

clk_action_list = []

temp_thes = 65

for i in range(3):
    for j in range(3):
        clk_action=(i,j)
        clk_action_list.append(clk_action)

def get_reward(fps, power, target_fps, c_t, g_t, c_t_prev, g_t_prev, beta):
	v1=0
	v2=0
	# print('power={}'.format(power))
	u=max(1,fps/target_fps)

	if g_t<= temp_thes:
		v2=0
	else:
		v2=2*(temp_thes-g_t)
	if c_t_prev < temp_thes:
		if c_t >= temp_thes:
			v1=-2

	if fps>=target_fps:
		u=1
	else :
		u=math.exp(0.1*(fps-target_fps))
	return u+v1+v2+beta/power

def action_to_freq(action):

    cpu_index = clk_action_list[action][0]
    gpu_index = clk_action_list[action][1]


    little_min, little_max = little_available_frequencies[cpu_index], little_available_frequencies[cpu_index]
    mid_min, mid_max = mid_available_frequencies[cpu_index], mid_available_frequencies[cpu_index]
    big_min, big_max = big_available_frequencies[cpu_index], big_available_frequencies[cpu_index]
    gpu_min, gpu_max = gpu_available_frequencies[gpu_index], gpu_available_frequencies[gpu_index]

    return little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max


class DVFStrain(Env):
    def __init__(self, initSleep, experiment, qos_type: str, targetTemp, latency):

        self.exp = experiment
        self.qos_type = qos_type
        self.latency = latency


        global temp_thes

        temp_thes = targetTemp

        # adb root
        set_root()

        turn_on_screen()

        set_brightness(158)

        self.window = get_window()
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
        
        unset_frequency()

        turn_off_usb_charging()

        set_rate_limit_us(1000000000, 2000000)

        # energy before
        t1a, t2a, littlea, mida, biga, gpua = get_energy()

        # set dvfs
        little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max = action_to_freq(8)
        set_frequency(little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max)
        self.qos_time_prev = time.time()
        self.byte_prev = None
        self.packet_prev = None
        match qos_type:
            case "byte":
                self.byte_prev = get_packet_info(self.window, qos_type)
            case "packet":
                self.packet_prev = get_packet_info(self.window, qos_type)
        
        # # wait 1s
        sleep(1)

        match qos_type:
            case "fps":
                qos = get_fps(self.window)
            case "byte":
                byte_cur = get_packet_info(self.window, qos_type)
                qos_time_cur = time.time()
                qos = cal_packet((self.byte_prev, byte_cur), (self.qos_time_prev, qos_time_cur))
                print(byte_cur[1] - self.byte_prev[1], byte_cur[0] - self.byte_prev[0], qos_time_cur - self.qos_time_prev, qos)
                self.byte_prev = byte_cur
                self.qos_time_prev = qos_time_cur
            case "packet":
                packet_cur = get_packet_info(self.window, qos_type)
                qos_time_cur = time.time()
                qos = cal_packet((self.packet_prev, packet_cur), (self.qos_time_prev, qos_time_cur))
                self.packet_prev = packet_cur
                self.qos_time_prev = qos_time_cur

        # energy after
        t1b, t2b, littleb, midb, bigb, gpub = get_energy()

        # reward - energy
        little = (littleb - littlea)/(t1b-t1a)
        mid = (midb - mida)/(t1b-t1a)
        big = (bigb - biga)/(t1b-t1a)
        gpu = (gpub - gpua)/(t2b-t2a)

        l_t, m_t, b_t, g_t, qi_t, disp_t  = get_temperatures()

        cpu_t = (l_t + m_t + b_t)/3

        self.state = np.array([3, 3, (little + mid + big)/100, gpu/100, cpu_t, g_t, qos])
        
        self.c_t_prev = cpu_t
        self.g_t_prev = g_t

        # no. of rounds
        self.rounds = 0
        
        # reward collected
        self.collected_reward = 0

        self.last_energy = t1b, t2b, littleb, midb, bigb, gpub
        self.last_util = get_core_util()
     
    def step(self, action):

        # set dvfs
        little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max = action_to_freq(action)
        set_frequency(little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max)

        # energy before
        # t1a, t2a, littlea, mida, biga, gpua = get_energy()
        t1a, t2a, littlea, mida, biga, gpua = self.last_energy

        # a = get_core_util()
        a = self.last_util

        # # wait 1s
        sleep(1)

        match self.qos_type:
            case "fps":
                qos = get_fps(self.window)
            case "byte":
                byte_cur = get_packet_info(self.window, self.qos_type)
                qos_time_cur = time.time()
                qos = cal_packet((self.byte_prev, byte_cur), (self.qos_time_prev, qos_time_cur))
                print(byte_cur[1] - self.byte_prev[1], byte_cur[0] - self.byte_prev[0], qos_time_cur - self.qos_time_prev, qos)
                self.byte_prev = byte_cur
                self.qos_time_prev = qos_time_cur
            case "packet":
                packet_cur = get_packet_info(self.window, self.qos_type)
                qos_time_cur = time.time()
                qos = cal_packet((self.packet_prev, packet_cur), (self.qos_time_prev, qos_time_cur))
                self.packet_prev = packet_cur
                self.qos_time_prev = qos_time_cur

        # energy after
        t1b, t2b, littleb, midb, bigb, gpub = get_energy()
        b = get_core_util()
        cpu_util = np.array(list(cal_core_util(b,a)))
        gpu_util = get_gpu_util()

        util_li = np.concatenate([cpu_util, gpu_util])

        # reward - energy
        little = (littleb - littlea)/(t1b-t1a)
        mid = (midb - mida)/(t1b-t1a)
        big = (bigb - biga)/(t1b-t1a)
        gpu = (gpub - gpua)/(t2b-t2a)


        l_t, m_t, b_t, g_t, qi_t, batt_t  = get_temperatures()

        cpu_t = (l_t + m_t + b_t) / 3

        reward = get_reward(qos, little + mid + big + gpu, target_fps, cpu_t, g_t, self.c_t_prev, self.g_t_prev, beta)

        self.c_t_prev = cpu_t
        self.g_t_prev = g_t

        ppw = qos/(little + mid + big + gpu)

        info = {"little": little_min, "mid": mid_min, "big": big_min, "gpu": gpu_min, "qos":qos, "power":little + mid + big + gpu, "reward":reward, "ppw":ppw, "temp": [l_t, m_t, b_t, g_t, qi_t, batt_t], "util" : util_li}

        ac = [little_min, mid_min, big_min, gpu_min]

        cpu_index = clk_action_list[action][0]
        gpu_index = clk_action_list[action][1]

        obs = np.array([cpu_index, gpu_index, (little + mid + big)/100, gpu/100, cpu_t, g_t, qos])

        self.collected_reward += reward

        self.rounds += 1
        
        self.render(ac, reward)
            
        self.state = obs

        self.last_energy = t1b, t2b, littleb, midb, bigb, gpub
        self.last_util = b

        return obs, reward, True, False, info
    
    def reset(self, seed, options):
        super().reset(seed=seed)


        # # energy before
        # t1a, t2a, littlea, mida, biga, gpua = get_energy()

        # # set dvfs
        # little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max = action_to_freq(8)
        # set_frequency(little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max)

        # # wait 1s
        # sleep(0.5)

        # fps = get_fps(self.window)

        # # energy after
        # t1b, t2b, littleb, midb, bigb, gpub = get_energy()

        # # reward - energy
        # little = (littleb - littlea)/(t1b-t1a)
        # mid = (midb - mida)/(t1b-t1a)
        # big = (bigb - biga)/(t1b-t1a)
        # gpu = (gpub - gpua)/(t2b-t2a)

        # l_t, m_t, b_t, g_t, qi_t, disp_t  = get_temperatures()

        # self.state = np.array([3, 3, (little + mid + big)/100, gpu/100, m_t, g_t, fps])
        
        # self.c_t_prev = m_t
        # self.g_t_prev = g_t

        # self.rounds = 0
        # self.collected_reward = 0
        return self.state, {"a":1}
    
    def render(self, action, rw):
        if self.rounds % 10 == 0:
            print(self.rounds, end = " ")
        # print(f"Round : {self.rounds}\nDistance Travelled : {np.round(action[0]/1000000, 3), np.round(action[1]/1000000, 3), np.round(action[2]/1000000, 3), np.round(action[3]/1000000, 3)}\nReward Received: {rw}")
        # print(f"Total Reward : {self.collected_reward}")
        # print("=============================================================================")