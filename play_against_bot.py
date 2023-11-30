from wizard import WizardGame, WizardState, card_to_action
from open_spiel.python.policy import Policy, UniformRandomPolicy
import numpy as np
import pickle

import pyspiel

def get_action_from_user(state: WizardState) -> int:
    legal_actions = state._legal_actions(state._next_player)
    print(list(map(lambda a: f'{a}: {state._action_to_string(state._next_player, a)}', legal_actions)))
    while True:
        try:
            res = int(input('Input an action: '))
            if res in legal_actions: return res
        except ValueError:
            pass

def get_action_fn_from_policy(policy: Policy):
    def get_action_from_state(state: WizardState):
        action_probs: dict[int, float] = policy.action_probabilities(state, state._next_player)
        actions, probs = list(action_probs.keys()), list(action_probs.values())
        return np.random.choice(actions, p=probs)
    return get_action_from_state

def main(action_fns: list):
    '''On each turn, calls the action_fn for the current player to get their action
    '''
    game: WizardGame = pyspiel.load_game('python_wizard')
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            #randomly sample a chance outcome
            actions, probs = zip(*state.chance_outcomes())
            action = np.random.choice(actions, p=probs)
        else:
            print('State: ' + str(state)+'\n\n')
            action = action_fns[state._next_player](state)
            print(f'Playing action: {state._action_to_string(state._next_player, action)}')
        state._apply_action(action)
    print(f'Game terminated with payouts: {state.returns()}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_file', type=str, default=None)
    args = parser.parse_args()
    game: WizardGame = pyspiel.load_game('python_wizard')
    bot_policy = UniformRandomPolicy(game)
    if args.policy_file is not None: 
        with open(args.policy_file) as f:
            bot_policy = pickle.load(f)
    main([get_action_from_user, get_action_fn_from_policy(UniformRandomPolicy(game))])