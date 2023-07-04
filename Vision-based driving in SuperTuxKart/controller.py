import pystk
import math

def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    # get the angle
    angle = aim_point[0]

    # break
    action.brake = False

    # acceleration & nitro
    if abs(angle) <= .1:
      action.nitro = True
    action.acceleration = 0 if current_vel > 20 else 1

    # drift & steer
    if abs(angle) <= .27 and abs(angle) > .1 and current_vel > 17.5:
      action.acceleration = 0.6
    elif abs(angle) > .27:
      action.drift = True
      if abs(angle) > .35 and current_vel > 13.5:
        action.acceleration = 0
        action.brake = True
    
    action.steer = min(1, angle * 3.7) if angle > 0 else max(-1, angle * 3.7)

    return action

if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
