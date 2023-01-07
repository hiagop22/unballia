
import onnxruntime
import hydra
import sys
import numpy as np
import clearml
import os
os.environ['CLEARML_CONFIG_FILE'] = '~/1_clearml_unball.conf'
from omegaconf import DictConfig
from PythonSimulator.field import Env
from pathlib import Path

class Test:
    def __init__(self, process_state, env: Env, max_v, max_w): 
        self.ort_session = None
        self.env = env
        self.process_state = process_state
        self.max_v = max_v
        self.max_w = max_w

    def load_model_from_clearml(self, task_id: str):
        task = clearml.Task.get_task(task_id=task_id)
        list_model = task.get_models()["output"][-1]
        ckpt_path = str(list_model.get_local_copy())
        
        # ckpt_path = Path(hydra.utils.get_original_cwd())  / 'last-model.onnx'

        self.ort_session = onnxruntime.InferenceSession(str(ckpt_path))
    
    def load_local_model(self, ckpt_path: Path):
        self.ort_session = onnxruntime.InferenceSession(str(ckpt_path))

    def run(self):
        state = self.env.reset_random_init_pos()
        done = False
        
        while not done and self.env.keep_running:
            ort_inputs = {self.ort_session.get_inputs()[0].name: [self.process_state.process(state)]}
            ort_outs = self.ort_session.run(None, ort_inputs)
            out = ort_outs[0]
            
            actions = np.random.normal(out)
            clipped_actions = np.clip(actions, -1.0, 1.0)
            env_action = np.reshape(clipped_actions, (-1,2))
            
            env_action[0] = env_action[0]*np.array([self.max_v,self.max_w])
            next_state, reward, done = self.env.step(env_action)
            state = next_state
            print('>>>reward')
            print(reward)

@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    
    process_state = hydra.utils.instantiate(cfg.process_state)
    env = hydra.utils.instantiate(cfg.env, render=True, random_impulse_ball2goal=False)
    test = Test(process_state, env=env, max_v=cfg.agent.max_v, max_w=cfg.agent.max_w)
    test.load_model_from_clearml(cfg.test.task_id)
    test.run()    

if __name__ == '__main__':
    sys.exit(main())