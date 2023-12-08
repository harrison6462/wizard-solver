from wizard import WizardGame, WizardObserver, WizardState, card_to_action
from open_spiel.python.policy import Policy, UniformRandomPolicy
from open_spiel.python.algorithms.cfr import CFRPlusSolver
import numpy as np
import pickle

import pyspiel
from open_spiel.python.algorithms import mcts
from dummy_mcts import *


def get_action_from_user(state: WizardState, player_id) -> int:
    legal_actions = state._legal_actions(player_id)
    print(list(map(lambda a: f'{a}: {state._action_to_string(player_id, a)}', legal_actions)))
    while True:
        try:
            res = int(input('Input an action: '))
            if res in legal_actions: return res
        except ValueError:
            pass



def get_action_fn_from_mcts():
    def get_action_from_state(state: WizardState, player_id):
        return do_action(state, player_id)
    return get_action_from_state

def get_action_fn_from_policy(policy: Policy):
    def get_action_from_state(state: WizardState, player_id):
   
        action_probs: dict[int, float] = policy.action_probabilities(state)
        actions, probs = list(action_probs.keys()), list(action_probs.values())
        return np.random.choice(actions, p=probs)
    return get_action_from_state

def main(action_fns: list, iters):
    '''On each turn, calls the action_fn for the current player to get their action
    '''
    points_diff = 0
    num_wins = 0
    num_draws = 0 
    for i in range(iters):
        try:
            game: WizardGame = pyspiel.load_game('python_wizard')
            state = game.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    #randomly sample a chance outcome
                    #change this to generating a random int and call the function you should cal (ask me about this gil)
                    actions, probs = zip(*state.chance_outcomes())
                    action = np.random.choice(actions, p=probs)
                    action_str = state.action_to_string(pyspiel.PlayerId.CHANCE, action)
                    #print("Sampled action: ", action_str)
                else:
                # print('State: ' + str(state)+'\n\n')
                    
                    action = action_fns[state._next_player](state, state._next_player)
                # print(f'Playing action: {state._action_to_string(state._next_player, action)}')
                state._apply_action(action)
            points = state.returns()
            print(f'Game terminated with payouts: {points}')
            
            points_diff +=  points[0]
            if (points[0] > 0):
                num_wins += 1
            elif points[0] == 0:
                num_draws +=1
        except Exception as E:
            print('ooi')
            print(E)
    return points_diff/(iters * 1.0), num_wins/(iters * 1.0), num_draws/(iters * 1.0)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_file', type=str, default=None)
    parser.add_argument('--player_1', type=str, default='human')
    parser.add_argument('--player_2', type=str, default='mcts')
    parser.add_argument("--iters", type=int, default=1)
    args = parser.parse_args()
    game: WizardGame = pyspiel.load_game('python_wizard')
    bot_policy = get_action_fn_from_policy(UniformRandomPolicy(game))
    if args.policy_file is not None: 
        with open(args.policy_file, 'rb') as f:
            bot_policy = pickle.load(f).tabular_average_policy()

    users = []
    if(args.player_1 == 'human'):
        users.append(get_action_from_user) 
    elif(args.player_1 == 'cfr'):
        users.append(get_action_fn_from_policy(bot_policy))
    elif(args.player_1 == 'mcts'):
        
        users.append(get_action_fn_from_mcts())
    elif(args.player_1 == 'random'):
        users.append(get_action_fn_from_policy(UniformRandomPolicy(game)))
    else:
        print('invalid player 1')
        raise Exception
    if(args.player_2 == 'human'):
        users.append(get_action_from_user) 
    elif(args.player_2 == 'cfr'):
        users.append(get_action_fn_from_policy(bot_policy))
    elif(args.player_2 == 'mcts'):
       
        users.append(get_action_fn_from_mcts())
    elif(args.player_2 == 'random'):
        users.append(get_action_fn_from_policy(UniformRandomPolicy(game)))
    else:
        print('invalid player 2')
        raise Exception
    
    points_diff, percent_win, percent_draw = main(users, args.iters)
    print('first player wins by ' + str(points_diff) + ' on average. wins ' + str(percent_win) + ' on average draw ' + str(percent_draw))
   
