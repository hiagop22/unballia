import torch
from torch import nn
from omegaconf import DictConfig
import hydra
from src.noise import OUNoise
import numpy as np
from typing import List
from torch.optim.optimizer import Optimizer
from .data import Experience, ReplayBuffer
from .utils import norm_grad, weights_init

device = torch.device('cuda:0')

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, activation = nn.Mish):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            activation(),
            nn.Linear(hidden_size,hidden_size),
            activation(),
            )

    def forward(self, x):
        return x + self.block(x)

class DDPGActor(nn.Module):
    """
    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment = (V,W)
    """
    def __init__(self, obs_size: int, n_actions: int, activation=nn.Mish):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_size, 512),
            activation(),
            ResidualBlock(hidden_size=512, activation=activation),
            ResidualBlock(hidden_size=512, activation=activation),
            nn.Linear(512, n_actions),
            nn.Tanh(),
            )

    def forward(self, x):
        x = self.layers(x)

        return x
        
class DDPGCritic(nn.Module):
    def __init__(self, obs_size: int,  n_actions: int, activation=nn.Mish):

        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(obs_size+n_actions, 512),
            activation(),
            ResidualBlock(512, activation=activation),
            ResidualBlock(512, activation=activation),
            nn.Linear(512, n_actions),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.layers(x)
        
        return x
        

class DDPGAgent(object):
    """Base agent class handeling the interaction with the environment"""
    """
    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """
    def __init__(self, env, replay_buffer: ReplayBuffer, process_state, max_v: float, max_w: float):
        
        self.env = env
        self.replay_buffer = replay_buffer
        self.process_state = process_state
        
        self.max_v = max_v
        self.max_w = max_w

        self.reset()
        
    def reset(self) -> None:
        # self.replay_buffer.reset()
        self.state = self.env.reset_random_init_pos()
        self.episode_reward = 0.0
        self.done = False
    
    @torch.no_grad()
    def play_episode(self, net: nn.Module, gamma: float, store_exp: bool = True) -> float:
        self.reset()
        
        action_space = 2
        noise = OUNoise(action_space)

        episode_reward = 0
        step = 0
        done = False

        while not done:
            action = self.get_action(net)
            env_action = np.reshape(action, (-1,2))

            env_action[0] = noise.get_action(env_action[0], step)
            env_action[0] = env_action[0]*np.array([self.max_v,self.max_w])
            
            new_state, reward, done = self.env.step(env_action)

            if store_exp:
                exp = Experience(self.process_state.process(self.state), action, reward, done, self.process_state.process(new_state))
                self.replay_buffer.append(exp)

            self.state = new_state
            episode_reward += reward
            
            step += 1

        if store_exp:
            self.replay_buffer.append(exp)
            self.replay_buffer.normalize_rewards(gamma)

        return episode_reward

    @torch.no_grad()
    def play_step(self, net: nn.Module, gamma: float, store_exp: bool = True) -> float:
        if self.done:
            self.reset()
        
        action_space = 2
        noise = OUNoise(action_space)

        episode_reward = 0
        step = 0
        done = False

        while not done:
            action = self.get_action(net)
            env_action = np.reshape(action, (-1,2))

            env_action[0] = noise.get_action(env_action[0], step)
            env_action[0] = env_action[0]*np.array([self.max_v,self.max_w])
            
            new_state, reward, done = self.env.step(env_action)

            if store_exp:
                exp = Experience(self.process_state.process(self.state), action, reward, done, self.process_state.process(new_state))
                self.replay_buffer.append(exp)

            self.state = new_state
            episode_reward += reward
            
            step += 1

        if store_exp:
            self.replay_buffer.append(exp)
            self.replay_buffer.normalize_rewards(gamma)

        return episode_reward
    
class DDPGAgent():
    def __init__(
        self,
        model_conf: DictConfig,
        memory_conf: DictConfig,
        optimizer_conf: DictConfig,
        scheduler_conf: DictConfig,
        process_state_conf: DictConfig, 
        max_grad_norm: DictConfig,
        gamma: float,
        tau: float,
        sync_max_dist_update: int,
        sync_target_network_update: int,
    ):

        self.actor = hydra.utils.instantiate(model_conf.actor).apply(weights_init)
        self.actor = self.actor.to(device)

        self.critic = hydra.utils.instantiate(model_conf.critic).apply(weights_init)
        self.critic = self.critic.to(device)

        self.process_state = hydra.utils.instantiate(process_state_conf)
        self.memory = hydra.utils.instantiate(memory_conf)

        self.scheduler_conf = scheduler_conf
        self.optimizer_conf = optimizer_conf
        
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        self.sync_max_dist_update = sync_max_dist_update
        self.sync_target_network_update = sync_target_network_update

        self.tau = tau

        self.target_actor = hydra.utils.instantiate(model_conf.actor).apply(weights_init)
        self.target_actor = self.target_actor.to(device)

        self.target_critic = hydra.utils.instantiate(model_conf.critic).apply(weights_init)
        self.target_critic = self.target_critic.to(device)

        self.hard_update_target_networks()

        self.actor_opt, self.critic_opt, self.actor_sch, self.critic_sch = self.configure_optimizers()

    def hard_update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
    
    def configure_optimizers(self) -> List[Optimizer]:
        actor_optimizer = hydra.utils.instantiate(
            self.optimizer_conf.actor, params=self.actor.parameters()
        )

        actor_scheduler = hydra.utils.instantiate(
            self.scheduler_conf, optimizer=actor_optimizer
        )

        critic_optimizer = hydra.utils.instantiate(
            self.optimizer_conf.critic, params=self.critic.parameters()
        )

        critic_scheduler = hydra.utils.instantiate(
            self.scheduler_conf, optimizer=critic_optimizer
        )

        return (actor_optimizer, critic_optimizer, actor_scheduler, critic_scheduler)

    def get_action(self, state):
        state = torch.tensor([self.process_state.process(state)], dtype=torch.float).to(device)
        action = self.actor(state)
       
        action = action.detach().cpu().numpy()
        action = np.clip(action, -1, 1)[0]

        return action

    def train_networks(self, batch_size):

        states, actions, rewards, dones, next_states = self.memory.sample(batch_size)

        # critic
        q_values = self.critic(states, actions)
        next_actions = self.target_actor(next_states)
        next_q = self.target_critic(next_states, next_actions)

        q_targets = rewards + self.gamma * next_q

        critic_loss = nn.MSELoss()(q_targets, q_values)
        self.critic_opt.zero_grad()
        self.manual_backward(critic_loss)
        norm_grad_critic = norm_grad(self.critic)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hparams.max_grad_norm.critic)
        self.critic_opt.step()

        # actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_opt.zero_grad()
        self.manual_backward(actor_loss)
        norm_grad_actor = norm_grad(self.actor)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.hparams.max_grad_norm.actor)
        self.actor_opt.step()

        self.soft_update_target_networks()

        output = {
            'critic_loss': critic_loss.detach().numpy(),
            'actor_loss': actor_loss.detach().numpy(),
            'norm_grad_actor': norm_grad_actor,
            'norm_grad_critic': norm_grad_critic,
        }

        return output

    def training_epoch_end(self):

        if not self.current_epoch % self.sync_max_dist_update and not self.current_epoch and self.env.max_dist < 0.75:
            self.env.max_dist += 0.10

        # val_reward = self.agent.play_episode(self.actor, self.gamma)

        # self.log(
        #     'val_reward',
        #     val_reward,
        #     on_epoch=True,
        #     prog_bar=False,
        #     logger=True,
        # )

        self.step_schedulers()

    def soft_update_target_networks(self):

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_experience(self, state, action, reward, done, new_state):

        exp = Experience(self.process_state.process(state), 
                        action, 
                        reward,
                        done, 
                        self.process_state.process(new_state),
                        )
        
        self.memory.append(exp)
       
    def step_schedulers(self):

        self.actor_sch.step()
        self.critic_sch.step()