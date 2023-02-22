from collections import deque, namedtuple
import numpy as np
import torch
from .utils import normalized_angle, angle_between
from torch.utils import data as tdata
import math

# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    'Experience',
    field_names=['state', 'action', 'reward', 'done', 'next_state'],
)


class ProcessStateV1:
    def __init__(self, state_size: int):
        self.state_size = state_size
    
    def process(self, state):
        robot2control = 0

        robot = state['ally'][robot2control]['pos_xy']
        angle_robot = state['ally'][robot2control]['theta']
        ball = state['ball']['pos_xy']
        v_robot = state['ally'][robot2control]['vel_xy']
        v_ball = state['ball']['vel_xy']
        robot_vel_unit = np.array([math.cos(angle_robot), math.sin(angle_robot)])
        # Remove the number later and use constant
        goal = np.array([0.75,0])

        robot_ball = ball - robot
        norm_robot_ball = np.linalg.norm(robot_ball)
        robot_ball_unit = robot_ball/(norm_robot_ball + 10e-5)
        proj_robot_robotball = np.dot(robot_ball_unit, robot_vel_unit)

        robot_goal = goal - robot
        norm_robot_goal = np.linalg.norm(robot_goal)
        robot_goal_unit = robot_goal /(norm_robot_goal + 10e-5)
        proj_robot_robotgoal = np.dot(robot_goal_unit, robot_vel_unit)

        vel_robot_ball = v_ball - v_robot
        norm_vel_robot_ball = np.linalg.norm(vel_robot_ball)
        vel_robot_ball_unit = vel_robot_ball/(norm_vel_robot_ball + 10e-5)
        proj_vel_robot_ball = np.dot(vel_robot_ball_unit, robot_vel_unit)

        processed_state = [norm_robot_ball, proj_robot_robotball, 
                            norm_robot_goal, proj_robot_robotgoal,
                            norm_vel_robot_ball, proj_vel_robot_ball,
                            ] 
        
        return processed_state
        

class AbsoluteState:
    def __init__(self, state_size: int):
        self.state_size = state_size

    def process(self, state):
        robot2control = 0

        robot = state['ally'][robot2control]['pos_xy']
        angle_robot = state['ally'][robot2control]['theta']
        v_robot = state['ally'][robot2control]['vel_xy']
        w_robot = state['ally'][robot2control]['w']
        ball = state['ball']['pos_xy']
        v_ball = state['ball']['vel_xy']
        robot_vel_unit = np.array([math.cos(angle_robot), math.sin(angle_robot)])
        # Remove the number later and use constant
        goal = np.array([0.75,0])

        processed_state = [ robot[0], robot[1],
                            ball[0], ball[1],
                            v_robot[0], v_robot[1],
                            v_ball[0],v_ball[1],
                            goal[0], goal[1],
                            -goal[0], -goal[1],
                            angle_robot, w_robot,
                            ]

        return processed_state

class RelativeState:
    def __init__(self, state_size: int):
        self.state_size = state_size

        self.len_field = np.array([1.5, 1.3])
        self.diagonal_len_field = np.sqrt(np.sum(self.len_field**2))

    def process(self, state):
        robot2control = 0

        robot = state['ally'][robot2control]['pos_xy']
        angle_robot = state['ally'][robot2control]['theta']
        v_robot = state['ally'][robot2control]['vel_xy']
        w_robot = state['ally'][robot2control]['w']

        robot_vel_unit = np.array([math.cos(angle_robot), math.sin(angle_robot)])

        ball = state['ball']['pos_xy']
        v_ball = state['ball']['vel_xy']
        
        opponent_goal = np.array([0.75,0.0])
        ally_goal = -opponent_goal

        robot_ball = ball - robot
        norm_robot_ball = np.linalg.norm(robot_ball)
        velrobot_robotballangle = angle_between(robot_vel_unit, robot_ball)

        ball_opponentgoal = opponent_goal - ball
        ball_allygoal = ally_goal - ball

        robot_opponentgoal = opponent_goal - robot
        norm_robot_opponentgoal = np.linalg.norm(robot_opponentgoal)

        robot_allygoal = ally_goal - robot
        norm_robot_allygoal = np.linalg.norm(robot_allygoal)

        vrobot_ballopponentgoal_angle = angle_between(robot_vel_unit, ball_opponentgoal)
        vrobot_ballallygoal_angle = angle_between(robot_vel_unit, ball_allygoal)

        norm_vel_robot = np.linalg.norm(v_robot)

        # Wall: 0 is far. 1 is near from one of them
        near_wall1 = (-robot[0] + 0.75)
        near_wall2 = (robot[0] + 0.75)
        near_wall3 = (-robot[1] + 0.65)
        near_wall4 = (robot[1] + 0.65)

        processed_state = [ velrobot_robotballangle, norm_robot_ball,
                            # norm_vel_robot, w_robot,
                            # angle_robot,
                            # near_wall1, near_wall2,
                            # near_wall3, near_wall4,
                            norm_robot_allygoal, norm_robot_opponentgoal,
                            vrobot_ballopponentgoal_angle, vrobot_ballallygoal_angle,
                            ] 
        
        return processed_state

class ReplayBuffer(object):
    """Replay buffer for storing past experiences allowing the agent to learn from them."""
    def __init__(self, memory_size: int) -> None:
        super().__init__()
        self.buffer = []
        self.memory = deque(maxlen=memory_size)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer
        
        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def normalize_rewards(self, gamma: float):
        # discounted_r = 0
        # disc_vector = []
        
        # for experience in reversed(self.buffer):
        #     discounted_r = discounted_r * gamma + experience.reward
        #     disc_vector.append(discounted_r)

        # disc_vector = np.array(disc_vector)
        # mean = disc_vector.mean()
        # std = disc_vector.std() + 1e-5

        discounted_r = 0
        for state, action, reward, done, new_state in reversed(self.buffer):
            discounted_r = discounted_r * gamma + reward
            
            # self.memory.append((state, action, (discounted_r - mean) / std, done, new_state))     
            self.memory.append((state, action, discounted_r, done, new_state))     

        
    def getitem(self, idx):
        return self.memory[idx]

    def reset(self):
        self.buffer = []
        # self.memory.clear()

class RLDataset(tdata.Dataset):
    """Iterable Dataset containing the ExperienceBuffer which will be used with the new experiences
    during training
    
    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, gamma: float=0.9) -> None:
        self.buffer = buffer
        self.gamma = gamma

    def __len__(self):
        return len(self.buffer.memory)

    def __getitem__(self, idx):
        state, action, reward, done, next_state = self.buffer.getitem(idx)

        return (
            torch.from_numpy(np.array(state)).float(),
            torch.from_numpy(np.array(action)).float(),
            torch.from_numpy(np.array([reward], dtype=np.float32)).float(),
            torch.from_numpy(np.array(done, dtype=np.bool)),
            torch.from_numpy(np.array(next_state)).float(),
        )

    def reset(self):
        self.buffer.reset()