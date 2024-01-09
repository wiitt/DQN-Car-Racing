import os
import matplotlib
import torch
import datetime
import csv

import gymnasium as gym
import gymnasium.wrappers as gym_wrap
import matplotlib.pyplot as plt
import numpy as np

from gymnasium.spaces import Box
from tensordict import TensorDict
from torch import nn
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info


class DQN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        cannel_n, height, width = in_dim

        if height != 84 or width != 84:
            raise ValueError(f"DQN model requires input of a (84, 84)-shape. Input of a ({height, width})-shape was passed.")

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=cannel_n, out_channels=16,
                      kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, input):
        return self.net(input)


class Agent:
    def __init__(
            self,
            state_space_shape,
            action_n,
            load_state=False,
            load_model=None,
            double_q=False,
            gamma = 0.95,
            epsilon = 1,
            epsilon_decay = 0.9999925,
            epsilon_min = 0.05
            ):
        
        self.gamma = gamma # discounting factor
        self.epsilon = epsilon # exploration rate
        self.epsilon_decay = epsilon_decay # decay of exploration rate
        self.epsilon_min = epsilon_min # minimum value of exploration rate
        self.state_shape = state_space_shape
        self.action_n = action_n
        self.load_state = load_state # purpose of loading the model (training or evaluation)
        self.double_q = double_q # if True, Double DQN is used
        self.save_dir = './training/saved_models/'
        self.log_dir = './training/logs/'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.updating_net = DQN(self.state_shape, self.action_n).float()
        self.updating_net = self.updating_net.to(device=self.device)
        self.frozen_net = DQN(self.state_shape, self.action_n).float()
        self.frozen_net = self.frozen_net.to(device=self.device)
        self.optimizer = torch.optim.Adam(self.updating_net.parameters(),
                                          lr=0.0002)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.buffer = TensorDictReplayBuffer(
                storage=LazyMemmapStorage(
                    300000,
                    device=torch.device("cpu")))
        self.act_taken = 0
        self.n_updates = 0
        if load_state:
            if load_model == None:
                raise ValueError(f"Specify a model name for loading.")
            load_dir = self.save_dir
            self.load_model = load_model
            self.load(load_dir, load_model)
        

    def store(self, state, action, reward, new_state, terminated):
        self.buffer.add(TensorDict({
                    "state": torch.tensor(state),
                    "action": torch.tensor(action),
                    "reward": torch.tensor(reward),
                    "new_state": torch.tensor(new_state),
                    "terminated": torch.tensor(terminated)
                    }, batch_size=[]))

    def get_samples(self, batch_size):
        batch = self.buffer.sample(
            batch_size)
        states = batch.get('state').type(torch.FloatTensor).to(self.device)
        new_states = batch.get('new_state').type(torch.FloatTensor).to(self.device)
        actions = batch.get('action').squeeze().to(self.device)
        rewards = batch.get('reward').squeeze().to(self.device)
        terminateds = batch.get('terminated').squeeze().to(self.device)
        return states, actions, rewards, new_states, terminateds

    def take_action(self, state):
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_n)
        else:
            state = torch.tensor(
                state,
                dtype=torch.float32,
                device=self.device
                ).unsqueeze(0)
            action_values = self.updating_net(state)
            action_idx = torch.argmax(action_values, axis=1).item()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
        self.act_taken += 1
        return action_idx

    def update_net(self, batch_size):
        self.n_updates += 1
        states, actions, rewards, \
            new_states, terminateds = self.get_samples(batch_size)
        action_values = self.updating_net(states)
        td_est = action_values[np.arange(batch_size), actions]
        if self.double_q:
            with torch.no_grad():
                next_actions = torch.argmax(self.updating_net(new_states), axis=1)
                tar_action_values = self.frozen_net(new_states)
            td_tar = rewards + (1 - terminateds.float()) \
                * self.gamma*tar_action_values[np.arange(batch_size), next_actions]
        else:
            with torch.no_grad():
                tar_action_values = self.frozen_net(new_states)
            td_tar = rewards + (1 - terminateds.float()) * self.gamma*tar_action_values.max(1)[0]
        loss = self.loss_fn(td_est, td_tar)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return td_est, loss

    def save(self, save_dir, save_name):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + save_name + f"_{self.act_taken}.pt"
        torch.save({
            'upd_model_state_dict': self.updating_net.state_dict(),
            'frz_model_state_dict': self.frozen_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_buffer': self.buffer,
            'action_number': self.act_taken,
            'epsilon': self.epsilon
            }, save_path)
        print(f"Model saved to {save_path} at step {self.act_taken}")

    def load(self, load_dir, model_name):
        loaded_model = torch.load(load_dir+model_name)
        upd_net_param = loaded_model['upd_model_state_dict']
        frz_net_param = loaded_model['frz_model_state_dict']
        opt_param = loaded_model['optimizer_state_dict']
        self.updating_net.load_state_dict(upd_net_param)
        self.frozen_net.load_state_dict(frz_net_param)
        self.optimizer.load_state_dict(opt_param)
        if self.load_state == 'eval':
            self.updating_net.eval()
            self.frozen_net.eval()
            self.epsilon_min = 0
            self.epsilon = 0
        elif self.load_state == 'train':
            self.updating_net.train()
            self.frozen_net.train()
            self.act_taken = loaded_model['action_number']
            self.epsilon = loaded_model['epsilon']
        else:
            raise ValueError(f"Unknown load state. Should be either 'eval' or 'train'.")
        
    def write_log(
            self,
            date_list,
            time_list,
            reward_list,
            length_list,
            loss_list,
            epsilon_list,
            log_filename='default_log.csv'
            ):

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        rows = [['date']+date_list,
                ['time']+time_list,
                ['reward']+reward_list,
                ['length']+length_list,
                ['loss']+loss_list,
                ['epsilon']+epsilon_list]
        with open(self.log_dir+log_filename, 'w') as csvfile:  
            csvwriter = csv.writer(csvfile)    
            csvwriter.writerows(rows)


def plot_reward(episode_num, reward_list, n_steps):
    plt.figure(1)
    rewards_tensor = torch.tensor(reward_list, dtype=torch.float)
    if len(rewards_tensor) >= 11:
        eval_reward = torch.clone(rewards_tensor[-10:])
        mean_eval_reward = round(torch.mean(eval_reward).item(), 2)
        std_eval_reward = round(torch.std(eval_reward).item(), 2)
        plt.clf()
        plt.title(f'Episode #{episode_num}: {n_steps} steps, \
                  reward {mean_eval_reward}±{std_eval_reward}')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_tensor.numpy())
    if len(rewards_tensor) >= 50:
        reward_f = torch.clone(rewards_tensor[:50])
        means = rewards_tensor.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.ones(49)*torch.mean(reward_f), means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    if is_ipython:
        display.display(plt.gcf())
        display.clear_output(wait=True)
