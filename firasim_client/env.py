import random
import math
import numpy as np
from firasim_client.libs.kdtree import KDTree
from firasim_client.libs.firasim import (FIRASimCommand, FIRASimVision)
from firasim_client.libs.entities import Field
from firasim_client.libs.entities import (Robot, Ball, NUM_ALLIES, NUM_OPPONENTS)

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

        self.ball = Ball()
        self.ally_robots = [Robot(id=id, ally=True, color=team_color) 
                            for id in range(Robot.num_allies)]
        self.opponent_robots = [Robot(id=id, ally=False, color=opponent_color) 
                                for id in range(Robot.num_opponents)]
        
    def step(self, actions):
        for id in range(self.num_allies_in_field):
            self.lin_and_ang_speed[robot] = (np.clip(actions[robot][0], -self.max_v, self.max_v), 
                                            np.clip(actions[robot][1], -self.max_w, self.max_w))

        # super(Env, self).run()
        self.send_velocities()
        self.get_frame()
        self.update_entity_properties()
        self.current_step += 1

        return (self.process_state(), self.reward(), self.done())
        # return new_state, reward, done

    def process_state(self):
        state = {'allie': {}, 
                 'opponent': {}, 
                 'ball': {},
                 }
        
        for id in range(self.num_allies_in_field):
            state['allie'][id] = { 'pos_xy': np.array([self.ally_robots.pos[0],self.ally_robots.pos[1]]), 
                                   'theta': self.ally_robots.pos[2], 
                                   'v': np.array(self.ally_robots[id].velxy),
                                   'w': self.ally_robots[id].w}

        for id in range(self.num_opponents_in_field):
            state['opponent'][id] = { 'pos_xy': np.array([self.opponent_robots.pos[0],self.opponent_robots.pos[1]]), 
                                      'theta': self.opponent_robots.pos[2], 
                                      'v': np.array(self.opponent_robots[id].velxy),
                                      'w': self.opponent_robots[id].w}
        
        state['ball'] = {'pos_xy':np.array(self.ball.pos), 
                         'v': np.array(self.ball.velxy),
                         }
                                        
        return state

    def reset_random_init_pos(self):
        self.current_step = 0

        ball_pos = [x(), y()]

        init_pos = {"ball": ball_pos,
                    "allies": [],
                    "opponents": [],
                    }

        min_dist = 10
        places = KDTree()
        places.insert(ball_pos)

        for id in range(self.num_allies_in_field):
            pos = [x(), y()]
            while places.get_nearest(pos)[1] < min_dist:
                pos = [x(), y()]

            places.insert(pos)
            init_pos["allies"].append(pos)
        
        if self.num_allies_in_field < NUM_ALLIES:
            for id in range(self.num_allies_in_field, NUM_ALLIES):
                self.ally_robots[id].move_outside_field()

        for _ in range(self.num_opponents_in_field):
            pos = [x(), y()]
            while places.get_nearest(pos)[1] < min_dist:
                pos = [x(), y()]

            places.insert(pos)
            init_pos.append(pos)

        if self.num_opponents_in_field < NUM_OPPONENTS:
            for id in range(self.num_opponents_in_field, NUM_OPPONENTS):
                self.opponent_robots[id].move_outside_field()

        for id in range(self.num_allies_in_field):
            self.ally_robots[id].pos = [*init_pos["allies"], theta()]
        
        for id in range(self.num_opponents_in_field):
            self.opponent_robots[id].pos = [*init_pos["opponents"], theta()]

        self.ball.pos = init_pos["ball"]

        return self.process_state()



if __name__ == "__main__":
    mdp = MarkovDecisionProcess()
    state = mdp.reset_random_init_pos()