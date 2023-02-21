import hydra
import shutil
import os
import clearml
from src.utils import save_onnx
import pytorch_lightning as pl

@hydra.main(version_base='1.2', config_path='config', config_name='config')
def main(cfg):
    
    # if cfg.is_debug:
    #     cfg.trainer.max_epochs = 5
    #     cfg.env.max_time_per_episode = 5

    pl.seed_everything(cfg.seed)
    
    if cfg.is_debug:
        clearml.Task.init(project_name=cfg.project_name, task_name=cfg.task_name, auto_connect_frameworks={'pytorch': False})
    else:
        clearml.Task.init(project_name=cfg.project_name, task_name=cfg.task_name)

    if os.path.isdir(cfg.logdir):
        shutil.rmtree(cfg.logdir)
        os.makedirs(cfg.logdir)
        
    logger = pl.loggers.TensorBoardLogger(cfg.logdir)
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
                dirpath=logger.save_dir,
                filename='best-model',
                save_last=True,
                verbose=True,
                monitor=cfg.watch_metric.actor.watch_metric,
                mode=cfg.watch_metric.actor.watch_metric_mode,
            )
    ]
        

    strategy_model = hydra.utils.instantiate(
        cfg.strategy,
        env_conf=cfg.env,
        model_conf=cfg.model,
        agent_conf=cfg.agent,
        buffer_conf=cfg.replaybuffer,
        optimizer_conf=cfg.optimizer,
        scheduler_conf=cfg.scheduler,
        dataset_conf=cfg.dataset,
        dataloader_conf=cfg.dataloader,
        process_state_conf=cfg.process_state,
        watch_metric=cfg.watch_metric,
        save_path=logger.save_dir,
        _recursive_=False,
        )
    
    # if cfg.pre_trained:
    #     task = clearml.Task.get_task(task_id=cfg.pre_trained_id)
    #     list_model = task.get_models()["output"][-1]
    #     ckpt_path = str(list_model.get_local_copy())

    #     while not ckpt_path.endswith('last.ckpt'):
    #         ckpt_path = str(list_model.get_local_copy())

    #     strategy_model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda' if torch.cuda.is_available() 
    #                                                     else 'cpu'))['state_dict'])
    
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)

    try: 
        trainer.fit(strategy_model)
    except KeyboardInterrupt:
        print('KeyboardInterrupt raised.')
    finally:
        if not cfg.is_debug:
            save_onnx(strategy_model, logger.save_dir, cfg.process_state.state_size)
        
if __name__ == '__main__':
    main()