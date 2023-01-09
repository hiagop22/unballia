from typing import List

NUM_ALLIES = 3
NUM_OPPONENTS = 3

class Goal:
    height: float = 0.4
    depth: float = 0.1

    def __init__(self):
        self.opponent_goal_pos: float = None
        self.ally_goal_pos: float = None


class Field(Goal):    
    height: float = 1.3
    width: float = 1.5
    
    def __init__(self, team_color: str = "blue"):
        super().__init__()
        self.team_color = team_color
        
        if team_color == "blue":
            self.opponent_goal_pos = Field/2
        elif team_color == "yellow":
            self.opponent_goal_pos = -Field/2

        self.ally_goal_pos = - self.opponent_goal_pos

# GOAL_AREA_WIDTH = 0.7
# GOAL_AREA_DEPTH = 0.15

class Robot:
    size: float = 0.075
    wheel_radius: float = 0.025
    max_velxy_norm: float = 2.0

    # [x,y,th]
    infinite_pos = {
        "ally": [[Field.width, Field.height, 0], 
                 [Field.width + 10, Field.height + 10, 0], 
                 [Field.width + 20, Field.height + 20, 0],
                ],

        "opponent": [[-Field.width, Field.height, 0], 
                     [-Field.width + 10, Field.height + 10, 0], 
                     [-Field.width + 20, Field.height + 20, 0],
                    ]
        }

    def __init__(self, id: int, ally: bool = True, color: str = "blue"):
        self.id = id
        self.pos = [0.0, 0.0, 0.0]
        self.velxy = [0.0, 0.0]
        self.w = 0.0

        self.type = "ally" if ally else "opponent"
        self.color = color
    
    def move_outside_field(self):
        self.pos = Robot.infinite_pos[self.type][self.id]

class Ball:
    radius: float = 0.02135

    def __init__(self):
        self.pos = [0, 0]
        self.velxy = [0, 0]