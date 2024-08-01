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


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():

        env = DVFStrain(args.initSleep, args.experiment)
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
        self.fc1 = nn.Linear(31, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc_mean = nn.Linear(10, 8)
        self.fc_logstd = nn.Linear(10, 8)
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
        log_std = -5 + 0.5 * (2 +5) * (log_std + 1)  # From SpinUp / Denis Yarats

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
        return action


if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}__exp{args.experiment}__temp{args.temperature}"


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    actor = Actor(envs).to(device)


    from torch.utils.mobile_optimizer import optimize_for_mobile
    
    actor.load_state_dict(torch.load("save/"+"DVFStrain6__OURS6__1__1722433335__exp3__temp20_actor"+".pt", map_location=device))
    model = actor
    torchscript_model = torch.jit.script(model)
    torchscript_model_optimized = optimize_for_mobile(torchscript_model)
    torch.jit.save(torchscript_model_optimized, "model.pt")
    
    XXX = torch.jit.load("model.pt")

    # states = np.array([34.612766 , 57.61118  , 69.707664 , 93.645836 , 83.       ,
    #     3.7997386,  1.3318955,  7.994837 ,  2.2368195, 69.       ,
    #    69.       , 72.       , 63.       , 46.774    , 38.6      ,
    #    39.9      , 30.333334 , 47.533333 ,  8.366667 ,  0.       ,
    #     0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
    #     9.       ,  0.       ,  5.       ,  0.       , 11.       ,
    #     6.       ], dtype=np.float32)
    
    states = np.array([25.26459  , 35.3488   , 35.204082 , 68.40853  ,  0.98     ,
        1.2124583,  0.5500983,  5.349312 ,  2.2273705, 67.       ,
       66.       , 68.       , 62.       , 47.419    , 39.4      ,
       10.       , 34.133335 , 36.866665 ,  6.733333 ,  0.       ,
       12.       ,  3.       ,  9.       ,  8.       , 18.       ,
        0.       ,  0.       ,  5.       ,  0.       , 11.       ,
        6.       ], dtype=np.float32)

    # env setup
    states = torch.Tensor(states).to(device)

    mean, log_std =XXX.forward(states)

    std = log_std.exp()
    normal = torch.distributions.Normal(mean, std)
    x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
    y_t = torch.tanh(x_t)
    action = y_t * actor.action_scale + actor.action_bias

    real_action = np.floor(action.detach().cpu().numpy() - 0.00000001).astype(np.int32)