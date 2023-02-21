import time
from numpy.core.fromnumeric import mean
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from omegaconf import DictConfig
import hydra
import numpy as np
from typing import Tuple, List
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from .data import Experience, ReplayBuffer
from .utils import norm_grad, weights_init, Mish

device = torch.device('cuda:0')

class Actor(nn.Module):
    """
    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment = (V,W)
    """
    def __init__(self, obs_size: int, n_actions: int, activation=Mish):
        super().__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(obs_size, 512),
            activation(),
            nn.Linear(512, 256),
            activation(),
            )

        self.mu = nn.Sequential(
            nn.Linear(256, n_actions),
            nn.Tanh(),
        )

        self.std = nn.Sequential(
            nn.Linear(256, n_actions),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.hidden_layers(x)
        means = self.mu(x)
        stds = torch.add(self.std(x), 1e-5)

        return means, stds

class Critic(nn.Module):
    def __init__(self, obs_size: int,  activation=Mish):

        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(obs_size, 512),
            activation(),
            nn.Linear(512, 256),
            activation(),
            nn.Linear(256, 1),
        )

    def forward(self, state):

        return self.layers(state)
        

class Agent:
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

    def get_action(self, net: nn.Module) -> int:        
        state = torch.tensor([self.process_state.process(self.state)], dtype=torch.float).to(device)
        means, stds = net(state)
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


class PPOStrategy(pl.LightningModule):
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
        upload_onnx_sync: int,
        entropy_beta: float,
        gamma: float,
        tau: float,
        save_path: str,
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
        self.entropy_beta = entropy_beta
        self.max_grad_norm = max_grad_norm
        self.tau = tau
        self.upload_onnx_sync = upload_onnx_sync

        self.initially_fill_buffer()

        self.save_path = save_path
    def initially_fill_buffer(self) -> None:
        """Carries out several one entire episode in the environment to initially fill up the replay buffer with
        experiences.
        """
        episode_reward = self.agent.play_episode(self.actor, self.gamma)

        self.log(
            'val_reward',
            episode_reward,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            logger=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            continuous v,w unnormalized. 
        """
        norm_dists = self.actor(x)

        return norm_dists


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
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hparams.max_grad_norm.critic)
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
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.hparams.max_grad_norm.actor)
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

    def training_epoch_end(self, training_step_outputs):
        # critic_loss_step = [out['critic_loss'] for out in training_step_outputs]
        # actor_loss_step = [out['actor_loss'] for out in training_step_outputs]
        # norm_grad_actor = [out['norm_grad_actor'] for out in training_step_outputs]
        # norm_grad_critic = [out['norm_grad_critic'] for out in training_step_outputs]
        
        # avg_critic_loss = mean(critic_loss_step)
        # avg_actor_loss = mean(actor_loss_step)
        # avg_norm_grad_actor = mean(norm_grad_actor)
        # avg_norm_grad_critic = mean(norm_grad_critic)
        
        # self.log('avg_critic_epoch_loss', 
        #         avg_critic_loss, 
        #         on_epoch=True, 
        #         prog_bar=True,
        #         logger=True,
        #     )

        # self.log('avg_actor_epoch_loss', 
        #         avg_actor_loss, 
        #         on_epoch=True, 
        #         prog_bar=True,
        #         logger=True,
        #     )

        # self.log('avg_norm_grad_actor', 
        #         avg_norm_grad_actor, 
        #         on_epoch=True, 
        #         prog_bar=True,
        #         logger=True,
        #     )

        # self.log('avg_norm_grad_critic', 
        #         avg_norm_grad_critic, 
        #         on_epoch=True, 
        #         prog_bar=True,
        #         logger=True,
        #     )

        val_reward = self.agent.play_episode(self.actor, self.gamma)

        self.log(
            'val_reward',
            val_reward,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.step_schedulers()

        # if not self.current_epoch % self.hparams.upload_onnx_sync:
        #     self.save_onnx_model()

    
    # def save_onnx_model(self):
    #     save_onnx(self, self.save_path, self.process_state.state_size)
       
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