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
from .utils import Mish, weights_init, save_onnx, upload_data, soft_update, hard_update, norm_grad
from PythonSimulator.field import Env
from clearml import Task

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, activation = nn.Tanh):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            activation(),
            nn.Linear(hidden_size,hidden_size),
            activation(),
            )

    def forward(self, x):
        return x + self.block(x)

class Noise(nn.Module):
    """
    Exploration Noise to robustness the model
    """
    def __init__(self, mu: float, sigma: float):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, action):
        noise = torch.randn_like(action) * self.sigma + self.mu*torch.ones_like(action)

        return noise


class Actor(nn.Module):
    """
    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers
    """
    def __init__(self, obs_size: int, n_actions: int, activation=nn.Tanh, hidden_size: int=64, num_residuals: int=4):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            activation(),
            )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_size=hidden_size) for _ in range(num_residuals)]
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_size, n_actions),
            nn.Tanh(),
        )

        self.n_actions = n_actions
        
    def forward(self, x):
        x = self.input(x.float())
        x = self.residual_blocks(x)
        x = self.out(x)

        return x

class Critic(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, activation=nn.Tanh, hidden_size: int=64, num_residuals: int=4):
        super().__init__()
        
        self.state = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            activation(),
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_size=hidden_size, activation=activation) for _ in range(num_residuals)]
        )

        self.action = nn.Sequential(
            nn.Linear(n_actions, hidden_size),
            activation(),
        )

        self.state_action = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, 1),
            # activation(),
        )

    def forward(self, state, action):
        state_value = self.state(state.float())
        x_state = self.residual_blocks(state_value)

        action_value = self.action(action.float())
        x_action = self.residual_blocks(action_value)

        state_action_value = self.state_action(torch.cat([x_state, x_action], 1))

        return state_action_value

class A2CAgent:
    """Base agent class handeling the interaction with the environment"""
    """
    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """
    def __init__(self, env: Env, replay_buffer: ReplayBuffer, process_state, max_v: float, max_w: float, noise: Noise):

        self.env = env
        self.replay_buffer = replay_buffer
        self.process_state = process_state
        self.state = self.env.reset_random_init_pos()
        self.env.keep_running = True
        
        self.max_v = max_v
        self.max_w = max_w
        self.noise = noise

    def reset(self) -> None:
        """Resets the environment and updates the state"""
        self.replay_buffer.reset()
        self.state = self.env.reset_random_init_pos()
        self.env.keep_running = True

    def get_action(self, net: nn.Module, epsilon: float) -> int:
        """Using the given network, decide what action to carry out using a proximal policy.

        Args:
            net: A2C network
        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = np.random.randn(net.n_actions)
        
        else:
            state = torch.tensor([self.process_state.process(self.state)], dtype=torch.float)
            action = net(state)
            noised_action = action.detach().data.numpy()
        
            action = np.clip(noised_action, -1, 1)[0]

        return action
    
    @torch.no_grad()
    def play_episode(self, net: nn.Module, epsilon: float, gamma: float) -> float:
        """Carries out a entire episode between the agent and the environment.

        Args:
            net: A2C network

        Returns:
            episode_reward
        """
        episode_reward = 0.0
        self.reset()
        done = False
        while not done:
            action = self.get_action(net, epsilon)
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


class A2CStrategy(pl.LightningModule):
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
        noise_conf: DictConfig,
        watch_metric,
        gamma: float,
        max_grad_norm: DictConfig,
        upload_onnx_sync: int,
        tau: float,
        epsilon_decay: int,
        epsilon_start: float,
        epsilon_end: float,
        episode_init_decay: int,
        save_path: str,
    ):

        super().__init__()
        self.automatic_optimization = False

        self.save_hyperparameters()

        self.env = hydra.utils.instantiate(env_conf, render=False, max_steps_episode=600)

        # self.actor = hydra.utils.instantiate(model_conf.actor, activation=Mish)
        self.actor = hydra.utils.instantiate(model_conf.actor, activation=Mish)
        self.actor.apply(weights_init)
        self.actor_target = hydra.utils.instantiate(model_conf.actor, activation=Mish)
        hard_update(self.actor_target, self.actor)

        self.critic = hydra.utils.instantiate(model_conf.critic, activation=Mish)
        self.critic.apply(weights_init)
        self.critic_target = hydra.utils.instantiate(model_conf.critic, activation=Mish)
        hard_update(self.critic_target, self.critic)

        self.process_state = hydra.utils.instantiate(process_state_conf)
        self.buffer = hydra.utils.instantiate(buffer_conf)
        self.noise = hydra.utils.instantiate(noise_conf)
        self.agent = hydra.utils.instantiate(agent_conf,
                                            env=self.env, 
                                            replay_buffer=self.buffer,
                                            process_state=self.process_state,
                                            noise=self.noise)
        self.env = hydra.utils.instantiate(env_conf)

        self.scheduler_conf = scheduler_conf
        self.optimizer_conf = optimizer_conf
        self.dataset_conf = dataset_conf
        self.dataloader_conf = dataloader_conf
        self.watch_metric = watch_metric
        
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.tau = tau
        self.upload_onnx_sync = upload_onnx_sync

        self.epsilon_decay= epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.episode_init_decay = episode_init_decay

        self.initially_fill_buffer(epsilon=1.0)

        self.save_path = save_path
    def initially_fill_buffer(self, epsilon: float) -> None:
        """Carries out several one entire episode in the environment to initially fill up the replay buffer with
        experiences.
        """
        episode_reward = self.agent.play_episode(self.actor, epsilon, self.gamma)

        self.log(
            'val_reward',
            episode_reward,
            on_epoch=True,
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
        actions = self.actor(x)

        return actions

    def actor_loss(self, batch: Tuple[Tensor]) -> Tensor:
        states, actions, rewards, dones, next_states = batch
        loss = -self.critic(states, self.actor(states)).mean() + torch.norm(states[:,0, None])

        return loss


    def critic_mse_loss(self, batch: Tuple[Tensor]) -> List[Tensor]:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch
        value = self.critic(states, actions)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_values = self.critic_target(next_states, next_actions.detach())
            next_values[dones] = 0.0
            next_values = next_values.detach()

        td_target = torch.unsqueeze(rewards, 1) + next_values*self.hparams.gamma
        loss = nn.MSELoss()(td_target, value)
        return loss

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

        critic_loss = self.critic_mse_loss(batch)
        critic_optimizer.zero_grad()
        self.manual_backward(critic_loss)
        norm_grad_critic = norm_grad(self.critic)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hparams.max_grad_norm.critic)
        critic_optimizer.step()

        actor_loss = self.actor_loss(batch)
        actor_optimizer.zero_grad()
        self.manual_backward(actor_loss)
        norm_grad_actor = norm_grad(self.actor)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.hparams.max_grad_norm.actor)
        actor_optimizer.step()

        self.log(
            'actor_loss',
            actor_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        
        self.log(
            'critic_loss',
            critic_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        output = {
            'critic_loss': critic_loss.detach().numpy(),
            'actor_loss': actor_loss.detach().numpy(),
            'norm_grad_actor': norm_grad_actor,
            'norm_grad_critic': norm_grad_critic,
        }

        self.update_target_models()

        return output

    def training_epoch_end(self, training_step_outputs):
        critic_loss_step = [out['critic_loss'] for out in training_step_outputs]
        actor_loss_step = [out['actor_loss'] for out in training_step_outputs]
        norm_grad_actor = [out['norm_grad_actor'] for out in training_step_outputs]
        norm_grad_critic = [out['norm_grad_critic'] for out in training_step_outputs]
        
        avg_critic_loss = mean(critic_loss_step)
        avg_actor_loss = mean(actor_loss_step)
        avg_norm_grad_actor = mean(norm_grad_actor)
        avg_norm_grad_critic = mean(norm_grad_critic)
        
        self.log('avg_critic_epoch_loss', 
                avg_critic_loss, 
                on_epoch=True, 
                prog_bar=True,
                logger=True,
            )

        self.log('avg_actor_epoch_loss', 
                avg_actor_loss, 
                on_epoch=True, 
                prog_bar=True,
                logger=True,
            )

        self.log('avg_norm_grad_actor', 
                avg_norm_grad_actor, 
                on_epoch=True, 
                prog_bar=True,
                logger=True,
            )

        self.log('avg_norm_grad_critic', 
                avg_norm_grad_critic, 
                on_epoch=True, 
                prog_bar=True,
                logger=True,
            )

        val_reward = self.agent.play_episode(self.actor, self.epsilon, self.gamma)

        self.log(
            'val_reward',
            val_reward,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.log(
            'epsilon',
            self.epsilon,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.step_schedulers()

        if not self.current_epoch % self.hparams.upload_onnx_sync:
            self.save_onnx_model()

        if self.current_epoch > self.episode_init_decay:
            self.update_epsilon()

    def update_epsilon(self):
        self.epsilon = max(
                self.epsilon_end, 
                self.epsilon_start*np.exp(-self.epsilon_decay*(self.current_epoch - self.episode_init_decay)),
            )


    def update_target_models(self):
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)
    
    def save_onnx_model(self):
        save_onnx(self, self.save_path, self.process_state.state_size)
       
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