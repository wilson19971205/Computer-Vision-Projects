import numpy as np
import torch

import torchvision
import torchvision.transforms.functional as TF
from torchvision import transforms

from .locator import CNNRegressor, load_locator


class Team:
    agent_type = 'image'

    def __init__(self):
        self.team = None
        self.num_players = None

        # load model
        self.puck_locator = load_locator()
        self.puck_locator.eval()
        self.reverse_count = [0,0]
        self.turn = [0,0]
        self.turn_out = [0,0]

    def new_match(self, team: int, num_players: int) -> list:
        self.reverse_count = [0,0]
        self.turn = [0,0]
        self.turn_out = [0,0]
        self.team, self.num_players = team, num_players
        return ['tux'] * num_players


    def to_numpy(self, location):
        return np.float32([location[0], location[2]])


    # controller
    def controller(self, player, puck_pos, kart_pos, idx):

        # initialize
        action = {'acceleration': 0, 'brake': False, 'drift': False, 'rescue': False, 'steer': 0}

        # get parameters
        kart_front = self.to_numpy(player['kart']['front'])
        facing = player['kart']['front'][1] > kart_pos[1]
        puck_x, puck_y = puck_pos[0], puck_pos[1]
        kart_velocity = np.linalg.norm(player['kart']['velocity'])

        # control hit puck angle
        tune_x = 0.01
        if self.team == 0:
            if facing:
                if kart_pos[0] > 10:
                    puck_x += tune_x
                elif kart_pos[0] < -10:
                    puck_x -= tune_x
            else:
                if kart_pos[0] < 10 and kart_pos[0] > 0:
                    puck_x -= tune_x
                elif kart_pos[0] > -10 and kart_pos[0] < 0:
                    puck_x += tune_x
        else:
            if facing:
                if kart_pos[0] < 10 and kart_pos[0] > 0:
                    puck_x -= tune_x
                elif kart_pos[0] > -10 and kart_pos[0] < 0:
                    puck_x += tune_x
            else:
                if kart_pos[0] > 10:
                    puck_x += tune_x
                elif kart_pos[0] < -10:
                    puck_x -= tune_x

        # steering control
        if puck_x < 0:
            action['steer'] = -1
        else:
            action['steer'] = 1

        # control velocity
        if kart_velocity > 20:
            action['acceleration'] = 0
        else:
            action['acceleration'] = 1

        # drift & accleration
        if puck_x < -0.25 or puck_x > 0.25:
            action['acceleration'] = 0.2
            action['drift'] = True
        else:
            action['drift'] = False

        return action
        

    def act(self, player_state, player_image):

        action = [{'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0},
                  {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}]


        for i in range(2):

            kart_pos = self.to_numpy(player_state[i]['kart']['location'])

            if (kart_pos[0] > 38 or kart_pos[0] < -38 or kart_pos[1] < -57 or kart_pos[1] > 57) and (self.reverse_count[i] == 0 and self.turn_out[i] == 0):
                self.reverse_count[i] += 15
                self.turn_out[i] += 20

            if self.reverse_count[i] > 0:
                self.reverse_count[i] -= 1
                action[i]['acceleration'] = 0
                action[i]['brake'] = True
                if (kart_pos[0] > 0 and kart_pos[1] > 0) or (kart_pos[0] < 0 and kart_pos[1] < 0):
                    self.turn[i] = -1
                    action[i]['steer'] = 1
                else:
                    self.turn[i] = 1
                    action[i]['steer'] = -1

            elif self.turn_out[i] > 0:
                self.turn_out[i] -= 1
                action[i]['acceleration'] = 1
                action[i]['brake'] = False
                action[i]['steer'] = self.turn[i]

            else:
                puck_coord = self.puck_locator(TF.to_tensor(player_image[i])[None]).squeeze(0).detach().numpy()
                action[i] = self.controller(player_state[i], puck_coord, kart_pos, i)

        return action
