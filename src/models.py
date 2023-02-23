import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from omegaconf import DictConfig
import hydra
from src.noise import OUNoise
import numpy as np
from typing import Tuple, List
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from .data import Experience, ReplayBuffer
from .utils import norm_grad, weights_init
from abc import ABC, abstractmethod

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


class PPOActor(nn.Module):
    """
    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment = (V,W)
    """
    def __init__(self, obs_size: int, n_actions: int, activation=nn.Mish):
        super().__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(obs_size, 512),
            activation(),
            ResidualBlock(hidden_size=512)
            )

        self.mu = nn.Sequential(
            ResidualBlock(hidden_size=512),
            nn.Linear(512, n_actions),
            nn.Tanh(),
        )

        self.std = nn.Sequential(
            ResidualBlock(hidden_size=512),
            nn.Linear(512, n_actions),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.hidden_layers(x)
        means = self.mu(x)
        stds = torch.add(self.std(x), 1e-5)

        return means, stds

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


class PPOCritic(nn.Module):
    def __init__(self, obs_size: int,  activation=nn.Mish):

        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(obs_size, 512),
            activation(),
            ResidualBlock(512),
            nn.Linear(512, 1),
        )

    def forward(self, state):

        return self.layers(state)
        
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
        

class Agent(ABC):
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
        self.state = self.env.reset_random_init_pos()
        
        self.max_v = max_v
        self.max_w = max_w

    def reset(self) -> None:
        """Resets the environment and updates the state"""
        self.replay_buffer.reset()
        self.state = self.env.reset_random_init_pos()

    @abstractmethod
    def get_action(self, net: nn.Module) -> int:        
        pass
    
    @abstractmethod
    def play_episode(self, net: nn.Module, gamma: float) -> float:
        pass
    
class PPOAgent(Agent):
    """Base agent class handeling the interaction with the environment"""
    """
    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """
    def __init__(self, env, replay_buffer: ReplayBuffer, process_state, max_v: float, max_w: float):
        
        super().__init__(env, replay_buffer, process_state, max_v, max_w)

    def get_action(self, net: nn.Module) -> int:        
        state = torch.tensor([self.process_state.process(self.state)], dtype=torch.float).to(device)
        means, stds = net(state)
        print("means, stds")
        print(means)
        print(stds)

        dists = torch.distributions.Normal(means, stds)
        action = dists.sample().detach().cpu().numpy()
    
        action = np.clip(action, -1, 1)[0]

        return action

    @torch.no_grad()
    def play_episode(self, net: nn.Module, gamma: float) -> float:
        episode_reward = 0.0
        self.reset()
        done = False
        
        while not done:
            action = self.get_action(net)
            env_action = np.reshape(action, (-1,2))
            env_action[0] = env_action[0]*np.array([self.max_v,self.max_w])
            
            new_state, reward, done = self.env.step(env_action)
            exp = Experience(self.process_state.process(self.state), action, reward, done, self.process_state.process(new_state))

            self.replay_buffer.append(exp)
            self.state = new_state
            episode_reward += reward

        self.replay_buffer.append(exp)
        self.replay_buffer.normalize_rewards(gamma)

        return episode_reward

class DDPGAgent(Agent):
    """Base agent class handeling the interaction with the environment"""
    """
    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """
    def __init__(self, env, replay_buffer: ReplayBuffer, process_state, max_v: float, max_w: float):
        
        super().__init__(env, replay_buffer, process_state, max_v, max_w)
        self.reset()
        
    def reset(self) -> None:
        # self.replay_buffer.reset()
        self.state = self.env.reset_random_init_pos()
        self.episode_reward = 0.0
        self.done = False

    def get_action(self, net: nn.Module) -> int:        
        state = torch.tensor([self.process_state.process(self.state)], dtype=torch.float).to(device)
        action = net(state)
        print("actions")
        print(action)

        action = action.detach().cpu().numpy()
        action = np.clip(action, -1, 1)[0]

        return action
    
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
    
class Strategy(pl.LightningModule):
    def __init__(
        self,
        env_conf: DictConfig,
        model_conf: DictConfig,
        buffer_conf: DictConfig,
        agent_conf: DictConfig,
        optimizer_conf: DictConfig,
        scheduler_conf: DictConfig,
        dataset_conf: DictConfig,
        dataloader_conf: DictConfig,
        process_state_conf: DictConfig, 
        watch_metric,
        max_grad_norm: DictConfig,
        gamma: float,
        sync_max_dist_update:int,
    ):
        
        super().__init__()
        self.automatic_optimization = False

        self.save_hyperparameters()

        self.env = hydra.utils.instantiate(env_conf)

        self.actor = hydra.utils.instantiate(model_conf.actor).apply(weights_init)
        self.actor = self.actor.to(device)

        self.critic = hydra.utils.instantiate(model_conf.critic).apply(weights_init)
        self.critic = self.critic.to(device)

        self.process_state = hydra.utils.instantiate(process_state_conf)
        self.buffer = hydra.utils.instantiate(buffer_conf)

        self.agent = hydra.utils.instantiate(agent_conf,
                                            env=self.env, 
                                            replay_buffer=self.buffer,
                                            process_state=self.process_state,
                                            )
        self.env = hydra.utils.instantiate(env_conf)

        self.scheduler_conf = scheduler_conf
        self.optimizer_conf = optimizer_conf
        self.dataset_conf = dataset_conf
        self.dataloader_conf = dataloader_conf
        self.watch_metric = watch_metric
        
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        self.sync_max_dist_update = sync_max_dist_update

    def initially_fill_buffer(self) -> None:
        """Carries out several one entire episode in the environment to initially fill up the replay buffer with
        experiences.
        """
        self.agent.play_episode(self.actor, self.gamma)
    
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
        
        return ({
            'optimizer': actor_optimizer,
            'lr_scheduler': {
                            'scheduler': actor_scheduler,
                            'monitor': self.watch_metric.actor.watch_metric,
                            }
        },
        {
            'optimizer': critic_optimizer,
            'lr_scheduler': { 
                            'scheduler': critic_scheduler,
                            'monitor': self.watch_metric.critic.watch_metric,
                            }   
        })

    def training_epoch_end(self, training_step_outputs):

        if not self.current_epoch % self.sync_max_dist_update and not self.current_epoch and self.env.max_dist < 0.75:
            self.env.max_dist += 0.10

        val_reward = self.agent.play_episode(self.actor, self.gamma)

        self.log(
            'val_reward',
            val_reward,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.step_schedulers()
       
    def step_schedulers(self):
        actor_sch, critic_sch = self.lr_schedulers()
        
        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(actor_sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            actor_sch.step(self.trainer.callback_metrics[self.watch_metric.actor.watch_metric])
        else:
            actor_sch.step()
            
        if isinstance(critic_sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            critic_sch.step(self.trainer.callback_metrics[self.watch_metric.critic.watch_metric])
        else:
            critic_sch.step()

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = hydra.utils.instantiate(self.dataset_conf, 
                                            buffer=self.buffer,
                                            gamma=self.gamma)
        dataloader = hydra.utils.instantiate(self.dataloader_conf,
                                            dataset=dataset)
        
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

class PPOStrategy(Strategy):
    def __init__(
        self,
        env_conf: DictConfig,
        model_conf: DictConfig,
        buffer_conf: DictConfig,
        agent_conf: DictConfig,
        optimizer_conf: DictConfig,
        scheduler_conf: DictConfig,
        dataset_conf: DictConfig,
        dataloader_conf: DictConfig,
        process_state_conf: DictConfig, 
        watch_metric,
        max_grad_norm: DictConfig,
        entropy_beta: float,
        gamma: float,
        sync_max_dist_update: int,
    ):

        super().__init__(
                        env_conf = env_conf,
                        model_conf = model_conf,
                        buffer_conf = buffer_conf,
                        agent_conf = agent_conf,
                        optimizer_conf = optimizer_conf,
                        scheduler_conf = scheduler_conf,
                        dataset_conf = dataset_conf,
                        dataloader_conf = dataloader_conf,
                        process_state_conf = process_state_conf,
                        watch_metric = watch_metric,
                        max_grad_norm = max_grad_norm,
                        gamma = gamma,
                        sync_max_dist_update = sync_max_dist_update,
                        )

        self.entropy_beta = entropy_beta

        self.initially_fill_buffer()

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            continuous v,w unnormalized. 
        """
        norm_dists = self.actor(x)

        return norm_dists
    

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx):
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the min:ibatch recieved.

        Args:
            batch: current mini batch of replay data
            batch_idx: batch number

        Returns:
            Training loss and log metrics
        """

        # calculates training loss
        actor_optimizer, critic_optimizer = self.optimizers()

        states, actions, discounted_rewards, dones, next_states = batch

        # critic
        td_targets = discounted_rewards
        values = self.critic(states)

        critic_loss = nn.MSELoss()(td_targets, values)
        critic_optimizer.zero_grad()
        self.manual_backward(critic_loss)
        norm_grad_critic = norm_grad(self.critic)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hparams.max_grad_norm.critic)
        critic_optimizer.step()

        # actor
        advantage = td_targets - values
        means, stds = self.actor(states)
        norm_dists = torch.distributions.Normal(means, stds)
        log_probs = norm_dists.log_prob(actions)
        entropy = norm_dists.entropy().mean()

        # actor_loss = (-logs_probs*advantage.detach()).mean()
        actor_loss = (-log_probs*advantage.detach()).mean() - entropy*self.entropy_beta
        actor_optimizer.zero_grad()
        self.manual_backward(actor_loss)
        norm_grad_actor = norm_grad(self.actor)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.hparams.max_grad_norm.actor)
        actor_optimizer.step()

        self.log(
            'actor_loss',
            actor_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )        

        self.log(
            'critic_loss',
            critic_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.log(
            'norm_grad_actor',
            norm_grad_actor,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.log(
            'norm_grad_critic',
            norm_grad_critic,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        
        self.log(
            'entropy',
            entropy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        output = {
            'critic_loss': critic_loss.detach().cpu().numpy(),
            'actor_loss': actor_loss.detach().cpu().numpy(),
            'norm_grad_actor': norm_grad_actor,
            'norm_grad_critic': norm_grad_critic,
        }

        return output


class DDPGStrategy(Strategy):
    def __init__(
        self,
        env_conf: DictConfig,
        model_conf: DictConfig,
        buffer_conf: DictConfig,
        agent_conf: DictConfig,
        optimizer_conf: DictConfig,
        scheduler_conf: DictConfig,
        dataset_conf: DictConfig,
        dataloader_conf: DictConfig,
        process_state_conf: DictConfig, 
        watch_metric,
        max_grad_norm: DictConfig,
        gamma: float,
        tau: float,
        sync_max_dist_update: int,
    ):

        super().__init__(
                        env_conf = env_conf,
                        model_conf = model_conf,
                        buffer_conf = buffer_conf,
                        agent_conf = agent_conf,
                        optimizer_conf = optimizer_conf,
                        scheduler_conf = scheduler_conf,
                        dataset_conf = dataset_conf,
                        dataloader_conf = dataloader_conf,
                        process_state_conf = process_state_conf,
                        watch_metric = watch_metric,
                        max_grad_norm = max_grad_norm,
                        gamma = gamma,
                        sync_max_dist_update = sync_max_dist_update,
                        )

        self.tau = tau

        self.target_actor = hydra.utils.instantiate(model_conf.actor).apply(weights_init)
        self.target_actor = self.target_actor.to(device)

        self.target_critic = hydra.utils.instantiate(model_conf.critic).apply(weights_init)
        self.target_critic = self.target_critic.to(device)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.initially_fill_buffer()

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            continuous v,w unnormalized. 
        """
        actions = self.actor(x)

        return actions

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx):
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the min:ibatch recieved.

        Args:
            batch: current mini batch of replay data
            batch_idx: batch number

        Returns:
            Training loss and log metrics
        """

        # calculates training loss
        actor_optimizer, critic_optimizer = self.optimizers()

        states, actions, rewards, dones, next_states = batch

        # critic
        q_values = self.critic(states, actions)
        next_actions = self.target_actor(next_states)
        next_q = self.target_critic(next_states, next_actions)

        q_targets = rewards + self.gamma * next_q

        critic_loss = nn.MSELoss()(q_targets, q_values)
        critic_optimizer.zero_grad()
        self.manual_backward(critic_loss)
        norm_grad_critic = norm_grad(self.critic)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hparams.max_grad_norm.critic)
        critic_optimizer.step()

        # actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_optimizer.zero_grad()
        self.manual_backward(actor_loss)
        norm_grad_actor = norm_grad(self.actor)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.hparams.max_grad_norm.actor)
        actor_optimizer.step()

        self.log(
            'actor_loss',
            actor_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )        

        self.log(
            'critic_loss',
            critic_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.log(
            'norm_grad_actor',
            norm_grad_actor,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.log(
            'norm_grad_critic',
            norm_grad_critic,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        output = {
            'critic_loss': critic_loss.detach().cpu().numpy(),
            'actor_loss': actor_loss.detach().cpu().numpy(),
            'norm_grad_actor': norm_grad_actor,
            'norm_grad_critic': norm_grad_critic,
        }

        return output
    

    def training_epoch_end(self, training_step_outputs):

        super().training_epoch_end(training_step_outputs)

        self.update_target_networks()

    def update_target_networks(self):

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
