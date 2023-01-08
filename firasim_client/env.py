import random
import math
from firasim_client.libs.kdtree import KDTree
from firasim_client.libs.firasim import (FIRASimCommand, FIRASimVision)
from firasim_client.libs.entities import Field
from firasim_client.libs.entities import Robot

def x(): 
    return random.uniform(-Field.width/2 + 10, Field.width/2 - 10)

def y(): 
    return random.uniform(-Field.height/2 + 10, Field.height/2 - 10)

def theta(): 
    return random.uniform(0, 2*math.pi)



class MarkovDecisionProcess:
    def __init__(self, 
                 max_steps_per_episode: int = 100,         
                 team_color: str = "blue",
                 opponent_color: str = "yellow",
                 num_allies_in_field: int = 1,
                 num_opponents_in_field: int = 0,
                 ):

        self.num_allies_in_field = num_allies_in_field
        self.num_opponents_in_field = num_opponents_in_field
        
        self.vision = FIRASimVision()

        self.blue_command = FIRASimCommand()
        self.yellow_command = FIRASimCommand(team_yellow=True)

        self.current_step = 0
        self.max_steps = max_steps_per_episode

        self.ally_robots = [Robot(id=id, ally=True, color=team_color) 
                            for id in range(Robot.num_allies)]
        self.opponent_robots = [Robot(id=id, ally=False, color=opponent_color) 
                                for id in range(Robot.num_opponents)]
        
    def step(self, action):
        for robot in range(self.num_allies):
            self.lin_and_ang_speed[robot] = (np.clip(actions[robot][0], -self.max_v, self.max_v), 
                                            np.clip(actions[robot][1], -self.max_w, self.max_w))

        super(Env, self).run()
        
        self.current_step += 1

        return (self.process_state(), self.reward(), self.done())
        # return new_state, reward, done

    def process_state(self):
        return_dict = {'allie': {}, 'opponent': {}, 'ball': {}}
        
        # X, y positions
        pos_x_allies = [self.robots_allies[i].body.position[0] for i in range(self.num_allies)]
        pos_y_allies = [self.robots_allies[i].body.position[1] for i in range(self.num_allies)]
        
        pos_x_opponents = [self.robots_opponents[i].body.position[0] for i in range(self.num_opponents)]
        pos_y_opponents = [self.robots_opponents[i].body.position[1] for i in range(self.num_opponents)]

        # Angular and linear velocities
        v_allies = [self.robots_allies[i].body.linearVelocity for i in range(self.num_allies)]
        w_allies = [self.robots_allies[i].body.angularVelocity for i in range(self.num_allies)]

        v_opponents = [self.robots_opponents[i].body.linearVelocity for i in range(self.num_opponents)]
        w_opponents = [self.robots_opponents[i].body.angularVelocity for i in range(self.num_opponents)]   

        # Theta angles
        theta_allies = [self.robots_allies[i].body.angle for i in range(self.num_allies)]
        theta_opponents = [self.robots_opponents[i].body.angle for i in range(self.num_opponents)]
        
        # Numpy creates an array with zero dimension, vector wich is seen as a scalar. So,
        # to create an array wich 1 row of dimension use the bellow command
        # To see more detais look the shape of array before and after the command bellow
        # return_array = np.expand_dims(return_array, axis=0)

        for i in range(self.num_allies):
            return_dict['allie'][i] = {'pos_xy': np.array([pos_x_allies[i], pos_y_allies[i]]), 
                                        'theta': theta_allies[i], 
                                        'v': np.array([v_allies[i][0],v_allies[i][1]]),
                                        'w': w_allies[i]}

        for i in range(self.num_opponents):
            return_dict['opponent'][i] = {'pos_xy': np.array([pos_x_opponents[i], pos_y_opponents[i]]), 
                                        'theta': theta_opponents[i], 
                                        'v': np.array([v_opponents[i][0],v_opponents[i][1]]),
                                        'w': w_opponents[i]}
        
        return_dict['ball'] = {'pos_xy':np.array([self.ball.body.position[0], 
                                        self.ball.body.position[1]]), 
                                'v': np.array([self.ball.body.linearVelocity[0],
                                            self.ball.body.linearVelocity[1]])}
                                        
        return return_dict

    def reset_random_init_pos(self):
        self.current_step = 0

        ball_pos = [x(), y()]

        init_pos = [ball_pos]

        min_dist = 10
        places = KDTree()
        places.insert(ball_pos)

        for i in range(self.num_allies):
            pos = [x(), y()]
            while places.get_nearest(pos)[1] < min_dist:
                pos = [x(), y()]

            places.insert(pos)
            init_pos.append(pos)
        
        for i in range(self.num_opponents):
            pos = [x(), y()]
            while places.get_nearest(pos)[1] < min_dist:
                pos = [x(), y()]

            places.insert(pos)
            init_pos.append(pos)

        self.robots_allies = [Robot(self.world, self.team_color, num_robot=x,
                                start_position=init_pos[1 + x], angle=theta(), 
                                y_predefined=False) for x in range(self.num_allies)]
        
        self.robots_opponents = [Robot(self.world, not self.team_color, num_robot=x, 
                                start_position=init_pos[1 + self.num_allies + x], angle=theta(),y_predefined=False,
                                ) for x in range(self.num_opponents)]

        return self.process_state()
        
    




if __name__ == "__main__":
    object = env()