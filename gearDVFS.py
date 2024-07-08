import os
import os.path as path
import csv
import json
import time
import datetime
import pickle
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import subprocess

from utils import *
from time import sleep

import os,sys
import math
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gearDVFSmodel import DQN_v0, ReplayMemory, DQN_AB

from torch.utils.tensorboard import SummaryWriter

"""
RL Agents are responsible for train/inference
Available Agents:
1. Vanilla Agent
2. Agent with Action Branching
"""

# Agent with action branching without time context
class DQN_AGENT_AB():
	def __init__(self, s_dim, h_dim, branches, buffer_size, params):
		"""
		s_dim是输入向量的维度
		h_dim是隐藏层的温度
		branches是一个列表内的每一个数字代表该分支的Actions大小。
		"""
  		
		self.eps = 0.8
		self.actions = [np.arange(i) for i in branches]
		# Experience Replay(requires belief state and observations)
		self.mem = ReplayMemory(buffer_size)
		# Initi networks
		self.policy_net = DQN_AB(s_dim, h_dim, branches)
		self.target_net = DQN_AB(s_dim, h_dim, branches)
		
		# self.weights = params["policy_net"]
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.target_net.eval()

		self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())
		self.criterion = nn.SmoothL1Loss() # Huber loss
		
	def max_action(self, state):
		# actions for multidomains
		max_actions = []
		with torch.no_grad():
			# Inference using policy_net given (domain, batch, dim)
			q_values = self.policy_net(state)
			for i in range(len(q_values)):
				domain = q_values[i].max(dim=1).indices
				max_actions.append(self.actions[i][domain])
		return max_actions

	def e_gready_action(self, actions, eps):
		# Epsilon-Gready for exploration
		final_actions = []
		for i in range(len(actions)):
			p = np.random.random()
			if isinstance(actions[i],np.ndarray):
				if p < 1- eps:
					final_actions.append(actions[i])
				else:
					# randint in (0, domain_num), for batchsize
					final_actions.append(np.random.randint(len(self.actions[i]),size=len(actions[i])))
			else:
				if p < 1- eps:
					final_actions.append(actions[i])
				else:
					final_actions.append(np.random.choice(self.actions[i]))
		final_actions = [int(i) for i in final_actions]
		return final_actions

	def select_action(self, state):
		return self.e_gready_action(self.max_action(state),self.eps)

	def train(self, n_round, n_update, n_batch):
		# Train on policy_net
		losses = []
		self.target_net.train()
		train_loader = torch.utils.data.DataLoader(
			self.mem, batch_size=n_batch, shuffle=True, drop_last=True)
		length = len(train_loader.dataset)
		GAMMA = 1.0
	
		# Calcuate loss for each branch and then simply sum up
		for i, trans in enumerate(train_loader):
			loss = 0.0 # initialize loss at the beginning of each batch
			states, actions, next_states, rewards = trans
			with torch.no_grad():
				target_result = self.target_net(next_states)
			policy_result = self.policy_net(states)
			# Loop through each action domain
			for j in range(len(self.actions)):
				next_state_values = target_result[j].max(dim=1)[0].detach()
				expected_state_action_values = (next_state_values*GAMMA) + rewards.float()
				# Gather action-values that have been taken
				branch_actions = actions[j].long()
				state_action_values = policy_result[j].gather(1, branch_actions.unsqueeze(1))
				loss += self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
			losses.append(loss.item())
			self.optimizer.zero_grad()
			loss.backward()
			if i>n_update:
				break

			self.optimizer.step()
		return losses

	def sync_model(self):
		self.target_net.load_state_dict(self.policy_net.state_dict())


def cal_cpu_reward(cpu_utils,cpu_temps,cluster_num):
    lambda_value = 0.15
    # for cpu
    cpu_u_max,cpu_u_min = 0.85,0.75
    cpu_u_g = 0.8
    u,v,w = -0.2,0.21,0.1
    temp_thre = 60
    reward_value = 0.0
    cpu_t =cpu_temps[0]
    print('cpu',end=': ')
    for cpu_u in cpu_utils:
        if cpu_u < cpu_u_min and cpu_u > cpu_u_max:
            d =lambda_value
        else:
            d = u+v*math.exp(-(cpu_u-cpu_u_g)**2 / (w ** 2))
        if cpu_t < temp_thre:
            w = 0.2 * math.tanh(temp_thre-cpu_t)
        else:
            w = -2
        reward_value += d
        print(f"{d}",end=',')
    
    return reward_value/cluster_num
  
def cal_gpu_reward(gpu_utils,gpu_temps,num):
    lambda_value = 0.1
    # for cpu
    gpu_u_max,gpu_u_min = 0.85,0.75
    gpu_u_g = 0.8
    u,v,w = -0.05,0.051,0.1
    temp_thre = 60
    reward_value = 0
    print('gpu',end=': ')
    for gpu_u,gpu_t in zip(gpu_utils,gpu_temps):
        if gpu_u < gpu_u_min and gpu_u > gpu_u_max:
            d =lambda_value
        else:
            d = u+v*math.exp(-(gpu_u-gpu_u_g)**2 / (w ** 2))
        if gpu_t < temp_thre:
            w = 0.2 * math.tanh(temp_thre-gpu_t)
        else:
            w = -2
        reward_value += d
        print(f"{d}",end=',')
    return reward_value/num 

def get_ob_phone(a, aa):
    # State Extraction and Reward Calculation

	t1a, t2a, littlea, mida, biga, gpua = aa

	cpu_util = get_cpu_util()
	gpu_util = get_gpu_util()

	little_f, mid_f, big_f, gpu_f = get_frequency()
	little_t, mid_t, big_t, gpu_t, qi_t, batt_t = get_temperatures()

	gpu_freq = [gpu_f]
	cpu_freq = [little_f, mid_f, big_f]
	gpu_thremal= [gpu_t]
	cpu_thremal = np.array([mid_t])
	

	b = get_core_util()
	t1b, t2b, littleb, midb, bigb, gpub = get_energy()

	cpu_util = list(cal_core_util(b,a))

	power = (littleb + midb + bigb - littlea - mida - biga)/(t1b-t1a) + (gpub-gpua)/(t2b-t2a)
	power2 = get_battery_power()

	# 16个数据
	states = np.concatenate([gpu_util,cpu_util,gpu_freq,cpu_freq,gpu_thremal,cpu_freq,power2]).astype(np.float32)

	reward = cal_cpu_reward(cpu_util,cpu_thremal,8)
	reward += cal_gpu_reward(gpu_util,gpu_thremal,1)
	print()
	return states,reward, power, [little_t, mid_t, big_t, gpu_t, qi_t, batt_t]

def action_to_freq(action):

	little_min, little_max = little_available_frequencies[action[0]], little_available_frequencies[action[0]]
	mid_min, mid_max = mid_available_frequencies[action[1]], mid_available_frequencies[action[1]]
	big_min, big_max = big_available_frequencies[action[2]], big_available_frequencies[action[2]]
	gpu_min, gpu_max = gpu_available_frequencies[action[3]], gpu_available_frequencies[action[3]]

	return little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max

if __name__ == "__main__":
	
	N_S, N_A, N_B = 5, 3, 11

	# Test/Train Demo for DQN_AB
	print("Test/Train Demo for DQN_AB")
	agent = DQN_AGENT_AB(N_S, 15, [3,5], 11, None)
	EPS_START = 0.99
	EPS_END = 0.2
	EPS_DECAY = 1000
	n_update, n_batch = 5,4
	SYNC_STEP = 30
	N_S,  N_BUFFER = 18, 36000
	agent = DQN_AGENT_AB(N_S,8,[11,14,17,12],N_BUFFER,None)
	prev_state, prev_action = [None]*2
	record_count, test_count, n_round, g_step = [0]*4

	global_count = 0

	run_name = "gearDVFS__" + str(int(time.time()))
	writer = SummaryWriter(f"runs/{run_name}")

	little = []
	mid = []
	big = []
	gpu = []
	ppw = []
	ts = []
	fpsLi = []
	rewardLi = []
	powerLi = []
	lossLi = []
	tempLi = []

	# adb root
	set_root()
	
	turn_off_usb_charging()

	turn_on_screen()

	set_brightness(158)

	window = get_window()

	unset_frequency()

	sleep(600)

	set_rate_limit_us(1000000, 2000)

	a = get_core_util()
	aa = get_energy()
	sleep(0.1)

	while True:
		state, reward, power1, temps = get_ob_phone(a, aa)
		agent.eps = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * g_step / EPS_DECAY)
		action = agent.select_action(torch.from_numpy(state).unsqueeze(0))
		if record_count!=0:
			agent.mem.push(prev_state, prev_action, state, reward)
		prev_state, prev_action = state, action
			
		# set dvfs
		little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max = action_to_freq(action)
		set_frequency(little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max)

		a = get_core_util()
		aa = get_energy()

		sleep(0.1)

		record_count+=1
		global_count += 1

		little1, mid1, big1, gpu1 = state[[-4, -3, -2, 9]]
		fps1 = get_fps(window)

		little.append(little1)
		mid.append(mid1)
		big.append(big1)
		gpu.append(gpu1)
		ppw.append(fps1/power1)
		fpsLi.append(fps1)
		powerLi.append(power1)
		rewardLi.append(reward)
		tempLi.append(temps)

		if (record_count%5==0 and record_count!=0):

			# train loop
			losses = agent.train(n_round,n_update,n_batch)

			if global_count % 10 == 0 and global_count != 0:
				writer.add_scalar("losses/loss", losses[-1], global_count)
				writer.add_scalar("freq/little", np.array(little)[-10:].mean(), global_count)
				writer.add_scalar("freq/mid", np.array(mid)[-10:].mean(), global_count)
				writer.add_scalar("freq/big", np.array(big)[-10:].mean(), global_count)
				writer.add_scalar("freq/gpu", np.array(gpu)[-10:].mean(), global_count)
				writer.add_scalar("perf/ppw", np.array(ppw)[-10:].mean(), global_count)
				writer.add_scalar("perf/reward", np.array(rewardLi)[-10:].mean(), global_count)
				writer.add_scalar("perf/power", np.array(powerLi)[-10:].mean(), global_count)
				writer.add_scalar("perf/fps", np.array(fpsLi)[-10:].mean(), global_count)
				writer.add_scalar("temp/little", np.array(tempLi)[-10:, 0].mean(), global_count)
				writer.add_scalar("temp/mid", np.array(tempLi)[-10:, 1].mean(), global_count)
				writer.add_scalar("temp/big", np.array(tempLi)[-10:, 2].mean(), global_count)
				writer.add_scalar("temp/gpu", np.array(tempLi)[-10:, 3].mean(), global_count)
				writer.add_scalar("temp/qi", np.array(tempLi)[-10:, 4].mean(), global_count)
				writer.add_scalar("temp/battery", np.array(tempLi)[-10:, 5].mean(), global_count)

			# Reset initial states/actions to None
			prev_state,prev_action,record_count = None,None,0
			# save model
			n_round += 1
			if n_round % SYNC_STEP == 0: agent.sync_model()
		
		if global_count >= 1000:
			turn_on_usb_charging()
			unset_rate_limit_us()
			break