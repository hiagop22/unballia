import hydra
import numpy as np
import shutil
import os
import clearml
from src.utils import save_onnx
from numpy.core.fromnumeric import mean
import pytorch_lightning as pl
from src.noise import OUNoise
from collections import defaultdict

def report2clearml(task, training_step_outputs, idx):
    buffer = defaultdict(list)
    
    for out in training_step_outputs:
        for key, val in out.items():
            buffer[key].append(val)

    avg_buffer = dict()

    for key, list_val in buffer.items():
        avg_value = mean(list_val)
        avg_buffer[key] = avg_value

        task.get_logger().report_scalar(key, key, value=avg_value, iteration=idx)

    return avg_buffer

@hydra.main(version_base='1.2', config_path='config', config_name='config')
def main(cfg):

    pl.seed_everything(cfg.seed)
    
    if cfg.is_debug:
        task = clearml.Task.init(project_name=cfg.project_name, task_name=cfg.task_name, auto_connect_frameworks={'pytorch': False})
    else:
        task = clearml.Task.init(project_name=cfg.project_name, task_name=cfg.task_name)

    if os.path.isdir(cfg.logdir):
        shutil.rmtree(cfg.logdir)
        os.makedirs(cfg.logdir)
        
    logger = pl.loggers.TensorBoardLogger(cfg.logdir)
    # callbacks = [
    #     pl.callbacks.LearningRateMonitor(),
    #     pl.callbacks.ModelCheckpoint(
    #             dirpath=logger.save_dir,
    #             filename='best-model',
    #             save_last=True,
    #             verbose=True,
    #             monitor=cfg.watch_metric.actor.watch_metric,
    #             mode=cfg.watch_metric.actor.watch_metric_mode,
    #         )
    # ]
    env = hydra.utils.instantiate(cfg.env)

    agent = hydra.utils.instantiate(
        cfg.agent,
        model_conf=cfg.model,
        memory_conf=cfg.memory,
        optimizer_conf=cfg.optimizer,
        scheduler_conf=cfg.scheduler,
        process_state_conf=cfg.process_state,
        _recursive_=False,
        )
    
    try: 
        for epoch in range(cfg.max_epochs):
            training_step_outputs = []
            done = False

            noise = OUNoise(cfg.action_size)
            episode_reward = 0

            state = env.reset_random_init_pos()
            step = 0

            while not done:
                action = agent.get_action(state)
                env_action = np.reshape(action, (-1,2))

                env_action[0] = noise.get_action(env_action[0], step)
                env_action[0] = env_action[0]*np.array([env.max_v, env.max_w])

                new_state, reward, done = env.step(env_action)
                
                agent.save_experience(state, action, reward, done, new_state)

                if len(agent.memory) > cfg.batch_size:
                    out = agent.train_networks(cfg.batch_size)
                    training_step_outputs.append(out)

                state = new_state
                step += 1
                episode_reward += reward

                
            if not epoch % cfg.sync_max_dist_update and not epoch and env.max_dist < 0.75:
                env.max_dist += 0.10

            avg_values = report2clearml(task, training_step_outputs, epoch)
            agent.step_schedulers(actor_loss=avg_values["actor_loss"], critic_loss=avg_values["critic_loss"])

            task.get_logger().report_scalar("reward", 
                                            "reward", 
                                            value=episode_reward, 
                                            iteration=epoch,
                                            )
            
            task.get_logger().report_scalar("epoch", 
                                            "epoch", 
                                            value=epoch, 
                                            iteration=epoch,
                                            )


    except KeyboardInterrupt:
        print('KeyboardInterrupt raised.')
    finally:
        if not cfg.is_debug:
            save_onnx(agent.actor, logger.save_dir, cfg.process_state.state_size)
        
if __name__ == '__main__':
    main()