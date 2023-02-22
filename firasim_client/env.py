import time
import random
import math
import numpy as np
from firasim_client.libs.kdtree import KDTree
from firasim_client.libs.firasim import (FIRASimCommand, FIRASimVision)
from firasim_client.libs.entities import Field
from firasim_client.libs.entities import (Robot, Ball, NUM_ALLIES, NUM_OPPONENTS)

# from libs.kdtree import KDTree
# from libs.firasim import (FIRASimCommand, FIRASimVision)
# from libs.entities import Field
# from libs.entities import (Robot, Ball, Goal, NUM_ALLIES, NUM_OPPONENTS)

# STATE
# frame {
#   ball {
#     x: -0.3412065754536228
#     y: 2.2364817958349936e-14
#     z: 0.023986730723637755
#     vx: 0.32858511930231765
#     vy: 3.9613809489793997e-13
#   }
#   robots_yellow {
#     x: 0.6750000119205631
#     y: -5.778237363890647e-19
#     vx: 1.0817582546700097e-14
#     vy: -1.0640574057521921e-17
#     vorientation: -6.338921697899231e-18
#   }
#   robots_yellow {
#     robot_id: 1
#     x: 0.2000000030338942
#     y: 7.788085422881489e-18
#     vx: -2.765251067452072e-15
#     vy: -4.194749088298115e-18
#     vorientation: 2.7148453649174645e-17
#   }
#   robots_yellow {
#     robot_id: 2
#     x: 0.40000000591614376
#     y: -2.3243176453601377e-17
#     vx: 5.428271683433795e-15
#     vy: -5.2189403989847574e-18
#     vorientation: 1.3557592297091665e-17
#   }
#   robots_blue {
#     x: -0.6750000119209177
#     y: -7.022068409336438e-19
#     vx: -1.106084329611927e-14
#     vy: -4.590733094815845e-18
#     vorientation: 5.411269140342285e-18
#   }
#   robots_blue {
#     robot_id: 1
#     x: -1.3679146655676533e-06
#     y: -1.2928810176337392e-19
#     vx: 1.5093794751880937e-05
#     vy: -8.033216692086013e-18
#     vorientation: 1.4444668903341924e-15
#   }
#   robots_blue {
#     robot_id: 2
#     x: -0.40008806842713857
#     y: -1.8673197541989945e-17
#     vx: -0.00014130264520262705
#     vy: -2.757269538220376e-16
#     vorientation: -4.306746913799341e-14
#   }
# }
# field {
#   width: 1.3
#   length: 1.5
#   goal_width: 0.4
#   goal_depth: 0.1
# }

# IMPORTANT:
# REWARDS ARE USING JUST INFO ABOUT ROBOT 0. NOT ABOUT ALL ALLY ROBOTS
# ALLY GOAL SUPOSED TO BEE ALWAYES IN NEGATIVE X

def x(): 
    return random.uniform(-Field.width/2 + 0.10, Field.width/2 - 0.10)

def y(): 
    return random.uniform(-Field.height/2 + 0.10, Field.height/2 - 0.10)

def theta(): 
    return random.uniform(0, 2*math.pi)



class MarkovDecisionProcess:
    def __init__(self, 
                 max_time_per_episode: int = 60*5,
                 team_color: str = "blue",
                 opponent_color: str = "yellow",
                 num_allies_in_field: int = 1,
                 num_opponents_in_field: int = 0,
                 time_step: float = 0.02,
                 min_dist: float = 0.10,
                 max_dist: float = 0.15,
                 ):

        self.min_dist = min_dist
        self.max_dist = max_dist

        self.num_allies_in_field = num_allies_in_field
        self.num_opponents_in_field = num_opponents_in_field
        
        self.vision = FIRASimVision()
        self.ally_command = FIRASimCommand( team_yellow = True 
                                            if team_color == "yellow"
                                            else False
                                            )

        self.current_step = 0
        self.max_time_per_episode = max_time_per_episode
        self.time_step = time_step

        self.ball = Ball()
        self.ally_robots = [Robot(id=id, ally=True, color=team_color) 
                            for id in range(NUM_ALLIES)]
        self.opponent_robots = [Robot(id=id, ally=False, color=opponent_color) 
                                for id in range(NUM_OPPONENTS)]
        self.field = Field(team_color=team_color)
        
        self.previous_ball_potential = None

    def step(self, actions):

        self.send_velocities2firasim(actions)
        firasim_frame = self.get_firasim_frame()
        self.update_entity_properties(firasim_frame)
        self.current_step += 1

        return (self.process_state(), self.reward(), self.done())

    def send_velocities2firasim(self, actions):
        """actions: tuple [[V, W]]"""
        
        tmp_actions = []
        for id in range(self.num_allies_in_field):
            vels = Robot.vel_vw2vel_whells(*actions[id])
            vl, vr = vels["vl"], vels["vr"]
            tmp_actions.append((vl, vr))

        
        self.ally_command.writeMulti(tmp_actions)

    def get_firasim_frame(self):
        last_packet = None
        
        last_time = time.time()
        while True:
            if (time.time() - last_time) < self.time_step:
                packet = self.vision.read()
                if packet is not None:
                    last_packet = packet
            else:
                if last_packet == None:
                    last_time = time.time()
                else:
                    break

        return last_packet

    def update_entity_properties(self, message):

        if self.field.team_color == "blue":
            robots = message.frame.robots_blue
        elif self.field.team_color == "yellow":
            robots = message.frame.robots_yellow

        for robot in robots:
            self.ally_robots[robot.robot_id].pos = [robot.x, robot.y, robot.orientation]
            self.ally_robots[robot.robot_id].velxy = [robot.vx, robot.vy]
            self.ally_robots[robot.robot_id].w = robot.vorientation

        self.ball.pos = [message.frame.ball.x, message.frame.ball.y]
        self.ball.velxy = [message.frame.ball.vx, message.frame.ball.vy]

    def process_state(self):
        state = {'ally': {}, 
                 'opponent': {}, 
                 'ball': {},
                 }
        
        for id in range(self.num_allies_in_field):
            state['ally'][id] = { 'pos_xy': np.array([self.ally_robots[id].pos[0],self.ally_robots[id].pos[1]]), 
                                   'theta': self.ally_robots[id].pos[2] if self.ally_robots[id].pos[2] > np.pi 
                                            else self.ally_robots[id].pos[2] - 2*np.pi, 
                                   'vel_xy': np.array(self.ally_robots[id].velxy),
                                   'w': self.ally_robots[id].w}

        for id in range(self.num_opponents_in_field):
            state['opponent'][id] = { 'pos_xy': np.array([self.opponent_robots[id].pos[0],self.opponent_robots[id].pos[1]]), 
                                      'theta': self.opponent_robots[id].pos[2] if self.opponent_robots[id].pos[2] > np.pi
                                                else self.opponent_robots[id].pos[2] - 2*np.pi, 
                                      'vel_xy': np.array(self.opponent_robots[id].velxy),
                                      'w': self.opponent_robots[id].w}
        
        state['ball'] = {'pos_xy':np.array(self.ball.pos), 
                         'vel_xy': np.array(self.ball.velxy),
                         }
                                        
        return state

    def reset_random_init_pos(self):
        self.init_time = time.time()

        self.set_entities_positions()

        self.send_positions2fira()

        return self.process_state()

    def set_entities_positions(self):

        ball_pos = [x(), y()]

        init_pos = {"ball": ball_pos,
                    "allies": [],
                    "opponents": [],
                    }

        places = KDTree()
        places.insert(ball_pos)

        for id in range(self.num_allies_in_field):
            pos = [x(), y()]
            while places.get_nearest(pos)[1] < self.min_dist or places.get_nearest(pos)[1] > self.max_dist:
                pos = [x(), y()]

            places.insert(pos)
            init_pos["allies"].append(pos)
        
        if self.num_allies_in_field < NUM_ALLIES:
            for id in range(self.num_allies_in_field, NUM_ALLIES):
                self.ally_robots[id].move_outside_field()

        for _ in range(self.num_opponents_in_field):
            pos = [x(), y()]
            while places.get_nearest(pos)[1] < self.min_dist:
                pos = [x(), y()]

            places.insert(pos)
            init_pos["opponents"].append(pos)

        if self.num_opponents_in_field < NUM_OPPONENTS:
            for id in range(self.num_opponents_in_field, NUM_OPPONENTS):
                self.opponent_robots[id].move_outside_field()

        for id in range(self.num_allies_in_field):
            self.ally_robots[id].pos = [*init_pos["allies"][id], theta()]
        
        for id in range(self.num_opponents_in_field):
            self.opponent_robots[id].pos = [*init_pos["opponents"][id], theta()]

        self.ball.pos = init_pos["ball"]

    def send_positions2fira(self):
        team_yellow = True if self.field.team_color == "yellow" else False

        opponent_command = FIRASimCommand(team_yellow = not team_yellow)

        self.ally_command.setBallPos(self.ball.pos[0], self.ball.pos[1])

        for id in range(NUM_ALLIES):            
            self.ally_command.setPos(id, *self.ally_robots[id].pos)
            
        for id in range(NUM_OPPONENTS):
            opponent_command.setPos(id, *self.opponent_robots[id].pos)

    def reward(self):
        """
        It's a simplified reward, that give +1 if the agent do a goal and
        return -1 reward if the agent receive a goal
        """
        reward = 0

        # ALLY GOAL SUPOSED TO BEE ALWAYES IN NEGATIVE X
        
        if self.ball.pos[0] > self.field.opponent_goal_pos[0]:
            reward = 10
        elif self.ball.pos[0] < self.field.ally_goal_pos[0]:
            reward = -10  
        else:

            w_move = 0.2
            w_ball_grad = 0.8
            w_energy = 2e-4

            # Calculate ball potential
            grad_ball_potential = self.ball_grad()
            # Calculate Move ball
            move_reward = self.move_reward()
            # Calculate Energy penalty
            energy_penalty = self.energy_penalty()

            reward = (  w_move * move_reward + \
                        w_ball_grad * grad_ball_potential + \
                        w_energy * energy_penalty
                        ) 
            # reward = (  w_ball_grad * grad_ball_potential + \
            #             w_energy * energy_penalty
            #             ) 

        return reward

    def done(self):
        
        if (time.time() - self.init_time) > self.max_time_per_episode:
            return True
            
        return True if abs(self.ball.pos[0]) > abs(self.field.ally_goal_pos[0]) else False

    def energy_penalty(self):

        energy_penalty = 0

        for id in range(self.num_allies_in_field):
            linearVelocity = math.sqrt(self.ally_robots[id].velxy[0]**2 + self.ally_robots[id].velxy[1]**2)
            en_penalty_1 = abs(linearVelocity)
            en_penalty_2 = abs(self.ally_robots[id].w)
            energy_penalty -= (en_penalty_1 + en_penalty_2)

        return energy_penalty
    
    def move_reward(self):
        '''
        Cosine between the robot vel vector and the vector robot -> ball.
        This indicates rather the robot is moving towards the ball or not.
        '''

        ball_pos = np.array(self.ball.pos)
        robot_pos = np.array([self.ally_robots[0].pos[0], self.ally_robots[0].pos[1]])
        robot_velxy = np.array(self.ally_robots[0].velxy)
        robot_ball = ball_pos - robot_pos
        unit_robot_ball = robot_ball/np.linalg.norm(robot_ball)

        move_reward = np.dot(unit_robot_ball, robot_velxy)

        move_reward = np.clip(move_reward / Robot.max_velxy_norm, -1.0, 1.0)
        return move_reward

    def ball_grad(self):
        '''
        Cosine between the ball vel vector and the vector ball -> goal.
        This indicates rather the ball is moving towards the goal or not.
        '''

        ball_pos = np.array(self.ball.pos)
        opponent_goal_pos = np.array(self.field.opponent_goal_pos)
        ball_velxy = np.array(self.ball.velxy)
        ball_opponent_goal = opponent_goal_pos - ball_pos
        unit_ball_opponent_goal = ball_opponent_goal/np.linalg.norm(ball_opponent_goal)

        move_reward = np.dot(unit_ball_opponent_goal, ball_velxy)

        move_reward = np.clip(move_reward / Ball.max_velxy_norm, -1.0, 1.0)
        return move_reward

class MarkovDecisionProcessV2(MarkovDecisionProcess):
    """
    Removing negative rewards
    """
    def __init__(self, 
                 max_time_per_episode: int = 60*5,
                 team_color: str = "blue",
                 opponent_color: str = "yellow",
                 num_allies_in_field: int = 1,
                 num_opponents_in_field: int = 0,
                 time_step: float = 0.02
                 ):
    
        super().__init__(
                    max_time_per_episode = max_time_per_episode,
                    team_color = team_color,
                    opponent_color = opponent_color,
                    num_allies_in_field = num_allies_in_field,
                    num_opponents_in_field = num_opponents_in_field,
                    time_step = time_step,
                    )
        

    def reward(self):
        """
        It's a simplified reward, that give +1 if the agent do a goal and
        return -1 reward if the agent receive a goal
        """
        reward = 0

        # ALLY GOAL SUPOSED TO BEE ALWAYES IN NEGATIVE X
        
        if self.ball.pos[0] > self.field.opponent_goal_pos[0]:
            reward = 0.5*np.exp(6.5)
        elif self.ball.pos[0] < self.field.ally_goal_pos[0]:
            reward = 0.5*np.exp(-6.5)
        else:

            w_move = 0.2
            w_ball_grad = 0.8
            w_energy = 2e-1

            # Calculate ball potential
            grad_ball_potential = self.ball_grad()
            # Calculate Move ball
            move_reward = self.move_reward()
            # Calculate Energy penalty
            energy_penalty = self.energy_penalty()

            reward = (  0.5*np.exp(w_move * move_reward) + \
                        0.5*np.exp(w_ball_grad * grad_ball_potential) + \
                        0.5*np.exp(w_energy * energy_penalty)
                        ) 
            # reward = (  
            #             0.5*np.exp(w_ball_grad * grad_ball_potential) + \
            #             0.5*np.exp(w_energy * energy_penalty)
            #             ) 

        return reward
    
    def energy_penalty(self):

        en_penalty = -abs(self.ally_robots[0].w)

        return en_penalty


class MarkovDecisionProcessV3(MarkovDecisionProcess):
    """
    Reward function from https://peerj.com/articles/cs-718.pdf, pg 10.
    """
    def __init__(self, 
                 max_time_per_episode: int = 60*5,
                 team_color: str = "blue",
                 opponent_color: str = "yellow",
                 num_allies_in_field: int = 1,
                 num_opponents_in_field: int = 0,
                 time_step: float = 0.02
                 ):
    
        super().__init__(
                    max_time_per_episode = max_time_per_episode,
                    team_color = team_color,
                    opponent_color = opponent_color,
                    num_allies_in_field = num_allies_in_field,
                    num_opponents_in_field = num_opponents_in_field,
                    time_step = time_step,
                    )
        

    def reward(self):
        ball_pos = np.array(self.ball.pos)
        robot_pos = np.array([self.ally_robots[0].pos[0], self.ally_robots[0].pos[1]])
        robot_ball = ball_pos - robot_pos

        norm_dist_robot_ball = np.linalg.norm(robot_ball)

        ball_velxy = np.array(self.ball.velxy)
        robot_velxy = np.array(self.ally_robots[0].velxy)
        vel_robot_ball = ball_velxy - robot_velxy
        norm_vel_robot_ball = np.linalg.norm(vel_robot_ball)

        reward = np.exp(-norm_dist_robot_ball) + 0.5*np.exp() + 0.5*(1 - np.exp(-norm_vel_robot_ball)) + 50*self.ball_grad()

        return reward

    def ball_grad(self):
        '''
        Cosine between the ball vel vector and the vector ball -> goal.
        This indicates rather the ball is moving towards the goal or not.
        '''

        ball_pos = np.array(self.ball.pos)
        opponent_goal_pos = np.array(self.field.opponent_goal_pos)
        ball_velxy = np.array(self.ball.velxy)
        ball_opponent_goal = opponent_goal_pos - ball_pos
        unit_ball_opponent_goal = ball_opponent_goal/np.linalg.norm(ball_opponent_goal)

        move_reward = np.dot(unit_ball_opponent_goal, ball_velxy)

        move_reward = np.clip(move_reward / Ball.max_velxy_norm, -1.0, 1.0)
        return move_reward
