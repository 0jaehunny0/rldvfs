# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from zTTenv import DVFStrain
import matplotlib.pyplot as plt
from utils import *
import sys

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "zTT"
    """the id of the environment"""
    total_timesteps: int = 1201
    """total timesteps of the experiments"""
    learning_rate: float = 0.05
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 500
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1
    """the timesteps it takes to update the target network"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.0
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 150
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""

    experiment: int = 1
    """the type of experiment"""
    temperature: int = 20
    """the ouside temperature"""
    initSleep: int = 600
    """initial sleep time"""
    loadModel: str = "no"
    """the save path of model"""
    timeOut: int = 30 * 60
    """the end time"""
    qos: str = "fps"
    """Quality of Service type"""
    targetTemp: int = 65
    """target temperature"""
    latency: int = 0
    """additional latency for adb communication"""

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():

        env = DVFStrain(args.initSleep, args.experiment, args.qos, args.targetTemp, args.latency)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, env.single_action_space.n),
        )

    def forward(self, x):
        x = x.float()
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":

    epsilon = 1
    epsilon_decay = 0.08


    little = []
    mid = []
    big = []
    gpu = []
    ppw = []

    # fig = plt.figure(figsize=(12,14))
    # ax1 = fig.add_subplot(4, 1, 1)
    # ax2 = fig.add_subplot(4, 1, 2)
    # ax3 = fig.add_subplot(4, 1, 3)
    # ax4 = fig.add_subplot(4, 1, 4)

    ts = []
    fpsLi = []
    bytesLi = []
    packetsLi = []
    rewardLi = []
    powerLi = []
    lossLi = []
    tempLi = []

    l1Li = []
    l2Li = []
    l3Li = []
    l4Li = []
    m1Li = []
    m2Li = []
    b1Li = []
    b2Li = []
    guLi = []


    print("asdf")

    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}__exp{args.experiment}__temp{args.temperature}"

    qos_type = args.qos

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # target_temp = args.targetTemp

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    if len(args.loadModel) > 2:
        q_network.load_state_dict(torch.load("save/"+args.loadModel+"_q_network.pt", map_location=device))
        target_network.load_state_dict(torch.load("save/"+args.loadModel+"_target_network.pt", map_location=device))


    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):

        if time.time() - start_time > args.timeOut:
            break

        ts.append(global_step)

        # ALGO LOGIC: put action logic here
        # epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

        epsilon *= epsilon_decay

        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        little.append(infos["final_info"][0]["little"])
        mid.append(infos["final_info"][0]["mid"])
        big.append(infos["final_info"][0]["big"])
        gpu.append(infos["final_info"][0]["gpu"])
        ppw.append(infos["final_info"][0]["ppw"])
        qos = infos["final_info"][0]["qos"]
        power = infos["final_info"][0]["power"]
        reward = infos["final_info"][0]["reward"]
        temps = infos["final_info"][0]["temp"]
        util_li = infos["final_info"][0]["util"]

        l1Li.append(util_li[0])
        l2Li.append(util_li[1])
        l3Li.append(util_li[2])
        l4Li.append(util_li[3])
        m1Li.append(util_li[4])
        m2Li.append(util_li[5])
        b1Li.append(util_li[6])
        b2Li.append(util_li[7])
        guLi.append(util_li[8])


        match qos_type:
            case "fps":
                fpsLi.append(qos)
            case "byte":
                bytesLi.append(qos)
            case "packet":
                packetsLi.append(qos)
        powerLi.append(power)
        rewardLi.append(reward)
        tempLi.append(temps)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 10 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("freq/little", np.array(little)[-10:].mean(), global_step)
                    writer.add_scalar("freq/mid", np.array(mid)[-10:].mean(), global_step)
                    writer.add_scalar("freq/big", np.array(big)[-10:].mean(), global_step)
                    writer.add_scalar("freq/gpu", np.array(gpu)[-10:].mean(), global_step)
                    writer.add_scalar("perf/ppw", np.array(ppw)[-10:].mean(), global_step)
                    writer.add_scalar("perf/reward", np.array(rewardLi)[-10:].mean(), global_step)
                    writer.add_scalar("perf/power", np.array(powerLi)[-10:].mean(), global_step)
                    writer.add_scalar("perf/fps", np.array(fpsLi)[-10:].mean(), global_step)
                    writer.add_scalar("temp/little", np.array(tempLi)[-10:, 0].mean(), global_step)
                    writer.add_scalar("temp/mid", np.array(tempLi)[-10:, 1].mean(), global_step)
                    writer.add_scalar("temp/big", np.array(tempLi)[-10:, 2].mean(), global_step)
                    writer.add_scalar("temp/gpu", np.array(tempLi)[-10:, 3].mean(), global_step)
                    writer.add_scalar("temp/qi", np.array(tempLi)[-10:, 4].mean(), global_step)
                    writer.add_scalar("temp/battery", np.array(tempLi)[-10:, 5].mean(), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )
        else:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)
            # target_max = max(target_max)
        
        if global_step % 10 == 0:
            writer.add_scalar("freq/little", np.array(little)[-10:].mean(), global_step)
            writer.add_scalar("freq/mid", np.array(mid)[-10:].mean(), global_step)
            writer.add_scalar("freq/big", np.array(big)[-10:].mean(), global_step)
            writer.add_scalar("freq/gpu", np.array(gpu)[-10:].mean(), global_step)
            writer.add_scalar("perf/ppw", np.array(ppw)[-10:].mean(), global_step)
            writer.add_scalar("perf/reward", np.array(rewardLi)[-10:].mean(), global_step)
            writer.add_scalar("perf/power", np.array(powerLi)[-10:].mean(), global_step)
            writer.add_scalar("perf/fps", np.array(fpsLi)[-10:].mean(), global_step)
            writer.add_scalar("temp/little", np.array(tempLi)[-10:, 0].mean(), global_step)
            writer.add_scalar("temp/mid", np.array(tempLi)[-10:, 1].mean(), global_step)
            writer.add_scalar("temp/big", np.array(tempLi)[-10:, 2].mean(), global_step)
            writer.add_scalar("temp/gpu", np.array(tempLi)[-10:, 3].mean(), global_step)
            writer.add_scalar("temp/qi", np.array(tempLi)[-10:, 4].mean(), global_step)
            writer.add_scalar("temp/battery", np.array(tempLi)[-10:, 5].mean(), global_step)

            little_c, mid_c, big_c, gpu_c = get_cooling_state()
            writer.add_scalar("cstate/little", little_c, global_step)
            writer.add_scalar("cstate/mid", mid_c, global_step)
            writer.add_scalar("cstate/big", big_c, global_step)
            writer.add_scalar("cstate/gpu", gpu_c, global_step)

            writer.add_scalar("util/l1", np.array(l1Li)[-10:].mean(), global_step)
            writer.add_scalar("util/l2", np.array(l2Li)[-10:].mean(), global_step)
            writer.add_scalar("util/l3", np.array(l3Li)[-10:].mean(), global_step)
            writer.add_scalar("util/l4", np.array(l4Li)[-10:].mean(), global_step)
            writer.add_scalar("util/m1", np.array(m1Li)[-10:].mean(), global_step)
            writer.add_scalar("util/m2", np.array(m2Li)[-10:].mean(), global_step)
            writer.add_scalar("util/b1", np.array(b1Li)[-10:].mean(), global_step)
            writer.add_scalar("util/b2", np.array(b2Li)[-10:].mean(), global_step)
            writer.add_scalar("util/gu", np.array(guLi)[-10:].mean(), global_step)
            writer.add_scalar("util/little", (np.array(l1Li[-10:]).mean()+np.array(l2Li[-10:]).mean()+np.array(l3Li[-10:]).mean()+np.array(l4Li[-10:]).mean()) / 4, global_step)
            writer.add_scalar("util/mid", (np.array(m1Li[-10:]).mean()+np.array(m2Li[-10:]).mean()) / 2, global_step)
            writer.add_scalar("util/big", (np.array(b1Li[-10:]).mean()+np.array(b2Li[-10:]).mean()) / 2, global_step)



        lossLi.append(float(loss))

        # ax1.plot(ts, fpsLi, linewidth=1, color='pink')
        # ax1.axhline(y=30, xmin=0, xmax=500)
        # ax1.set_title('Frame rate (Target fps = 30) ')
        # ax1.set_ylabel('Frame rate (fps)')
        # ax1.set_xlabel('Time (s) ')
        # ax1.set_xticks([0, 100, 200, 300, 400, 500])
        # ax1.set_yticks([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
        # ax1.grid(True)

        # ax2.plot(ts, powerLi, linewidth=1, color='blue')
        # ax2.set_title('Power consumption')
        # ax2.set_ylabel('Power (mW)')
        # ax2.set_yticks([0, 2000, 4000, 6000, 8000])
        # ax2.set_xticks([0, 100, 200, 300, 400, 500])
        # ax2.set_xlabel('Time (s) ')
        # ax2.grid(True)
        
        # ax3.plot(ts, rewardLi, linewidth=1, color='orange')
        # ax3.set_ylabel('reward')
        # ax3.set_xticks([0, 100, 200, 300, 400, 500])
        # ax3.set_xlabel('Time (s) ')
        # ax3.grid(True)
        
        # ax4.plot(ts, lossLi, linewidth=1, color='black')
        # ax4.set_ylabel('Average loss')
        # ax2.set_yticks([0, 2000, 4000, 6000, 8000])
        # ax4.set_xticks([0, 100, 200, 300, 400, 500])
        # ax4.set_xlabel('Time (s) ')
        # ax4.grid(True)
        
        # plt.pause(0.1)
    turn_on_usb_charging()
    unset_rate_limit_us()
    envs.close()
    writer.close()


    torch.save(q_network.state_dict(), "save/"+run_name+"_q_network.pt")
    torch.save(target_network.state_dict(), "save/"+run_name+"_target_network.pt")
