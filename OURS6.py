# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
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
# from my_env import DogTrain
from OURSenv6 import DVFStrain 
from utils2 import *
import time

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

    # Algorithm specific arguments
    env_id: str = "DVFStrain6"
    """the environment id of the task"""
    total_timesteps: int = 1001
    """total timesteps of the experiments"""
    buffer_size: int = 1001
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 100
    """the batch size of sample from the reply memory"""
    learning_starts: int = 100
    """timestep to start learning"""
    policy_lr: float = 1e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 5e-5
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    experiment: int = 1
    """the type of experiment"""
    temperature: int = 20
    """the ouside temperature"""
    initSleep: int = 1
    """initial sleep time"""
    loadModel: str = "no"
    """the save path of model"""
    timeOut: int = 30 * 60
    """the end time"""
    qos: str = "fps"
    """Quality of Service"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():

        env = DVFStrain(args.initSleep, args.experiment, args.qos)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc_mean = nn.Linear(10, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(10, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}__exp{args.experiment}__temp{args.temperature}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    qos_type = args.qos

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.MultiDiscrete), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    if len(args.loadModel) > 2:
        actor.load_state_dict(torch.load("save/"+args.loadModel+"_actor.pt", map_location=device))
        qf1.load_state_dict(torch.load("save/"+args.loadModel+"_qf1.pt", map_location=device))
        qf2.load_state_dict(torch.load("save/"+args.loadModel+"_qf2.pt", map_location=device))
        qf1_target.load_state_dict(torch.load("save/"+args.loadModel+"_qf1_target.pt", map_location=device))
        qf2_target.load_state_dict(torch.load("save/"+args.loadModel+"_qf2_target.pt", map_location=device))

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    little = []
    mid = []
    big = []
    gpu = []
    ppw = []
    ts = []
    fpsLi = []
    bytesLi = []
    packetsLi = []
    rewardLi = []
    powerLi = []
    lossLi = []
    tempLi = []
    timeLi = []
    upLi = []
    downLi = []
    gpuLi = []

    l1Li = []
    l2Li = []
    l3Li = []
    l4Li = []
    m1Li = []
    m2Li = []
    b1Li = []
    b2Li = []
    guLi = []

    from collections import deque
    utilLi = np.zeros

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):

        if time.time() - start_time > args.timeOut:
            break

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = np.floor(actions.detach().cpu().numpy() - 0.00000001).astype(np.int32)

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
        times = infos["final_info"][0]["time"]
        up_rate = infos["final_info"][0]["uptime"]
        down_rate = infos["final_info"][0]["downtime"]
        gpu_rate = infos["final_info"][0]["gputime"]
        util_li = infos["final_info"][0]["util"]

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
        timeLi.append(times)
        upLi.append(up_rate)
        downLi.append(down_rate)
        gpuLi.append(gpu_rate)


        l1Li.append(util_li[0])
        l2Li.append(util_li[1])
        l3Li.append(util_li[2])
        l4Li.append(util_li[3])
        m1Li.append(util_li[4])
        m2Li.append(util_li[5])
        b1Li.append(util_li[6])
        b2Li.append(util_li[7])
        guLi.append(util_li[8])


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
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 10 == 0:
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)


        if global_step % 10 == 0:
            writer.add_scalar("freq/little", np.array(little)[-10:].mean(), global_step)
            writer.add_scalar("freq/mid", np.array(mid)[-10:].mean(), global_step)
            writer.add_scalar("freq/big", np.array(big)[-10:].mean(), global_step)
            writer.add_scalar("freq/gpu", np.array(gpu)[-10:].mean(), global_step)
            writer.add_scalar("perf/ppw", np.array(ppw)[-10:].mean(), global_step)
            writer.add_scalar("perf/reward", np.array(rewardLi)[-10:].mean(), global_step)
            writer.add_scalar("perf/power", np.array(powerLi)[-10:].mean()*100, global_step)
            match qos_type:
                case "fps":
                    writer.add_scalar("perf/fps", np.array(fpsLi)[-10:].mean(), global_step)
                case "byte":
                    writer.add_scalar("perf/bytes", np.array(bytesLi)[-10:].mean(), global_step)
                case "packet":
                    writer.add_scalar("perf/packets", np.array(packetsLi)[-10:].mean(), global_step)
            writer.add_scalar("temp/little", np.array(tempLi)[-10:, 0].mean(), global_step)
            writer.add_scalar("temp/mid", np.array(tempLi)[-10:, 1].mean(), global_step)
            writer.add_scalar("temp/big", np.array(tempLi)[-10:, 2].mean(), global_step)
            writer.add_scalar("temp/gpu", np.array(tempLi)[-10:, 3].mean(), global_step)
            writer.add_scalar("temp/qi", np.array(tempLi)[-10:, 4].mean(), global_step)
            writer.add_scalar("temp/battery", np.array(tempLi)[-10:, 5].mean(), global_step)
            writer.add_scalar("losses/time", np.array(timeLi)[-10:].mean(), global_step)
            writer.add_scalar("losses/up", np.array(upLi)[-10:].mean(), global_step)
            writer.add_scalar("losses/down", np.array(downLi)[-10:].mean(), global_step)
            writer.add_scalar("losses/gpu_time", np.array(gpuLi)[-10:].mean(), global_step)
            
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


    turn_on_usb_charging()
    unset_rate_limit_us()
    turn_off_screen()
    unset_frequency()
    envs.close()
    writer.close()



    torch.save(actor.state_dict(), "save/"+run_name+"_actor.pt")
    torch.save(qf1.state_dict(), "save/"+run_name+"_qf1.pt")
    torch.save(qf2.state_dict(), "save/"+run_name+"_qf2.pt")
    torch.save(qf1_target.state_dict(), "save/"+run_name+"_qf1_target.pt")
    torch.save(qf2_target.state_dict(), "save/"+run_name+"_qf2_target.pt")
    # actor2 = Actor(envs).to(device)
    # actor2.load_state_dict(torch.load("asdf.pt", map_location=device))
