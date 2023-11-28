from __future__ import annotations

from copy import copy
from enum import Enum
import wizard_cfr_cpp
from wizard import Deck, Card, Hand, Face, Suit, _NUM_PLAYERS, _NUM_CARDS_PER_PLAYER, _DECK, WizardGame, WizardObserver, WizardState, card_to_action, generate_all_possible_hands, is_valid_move
from open_spiel.python.algorithms.best_response import BestResponsePolicy
from open_spiel.python.policy import UniformRandomPolicy, Policy, TabularPolicy
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
from itertools import product, combinations
import math
import pyspiel

def write_subgame_strategy_onto_policy_for_player(blueprint_strategy: TabularPolicy, subgame_name: str, subgame_policy: TabularPolicy, player: int) -> None:
    '''This will be kind of hacky, but not too different from how open spiel would handle this normally.
        We have the two tabular subgames
    '''
    subgame: WizardGame = pyspiel.load_game(subgame_name)
    observer = subgame.make_py_observer()
    for state in subgame_policy.states:
        try:
            state_policy = blueprint_strategy.policy_for_key(observer.string_from(state, player))
            for action, value in subgame_policy.action_probabilities(state, player).items():
                state_policy[action] = value
                print('Updating state')
        except LookupError as e:
            print(e)

def get_all_hands_consistent_with_observations(state: WizardState, player_cards_to_see: set = set()) -> list[list[Hand]]:
    '''Returns all possible hands the player can have from deck consistent with
    the observations (ie., if the lead suit is clubs and they play a non-wizard/jester
    non-club then they cannot have clubs).
    This is to be able to compute all possible histories as required by subgame solving.
    In fancier words, this is returning the common knowledge closure of the given state,
    as defined in https://proceedings.neurips.cc/paper/2021/file/c96c08f8bb7960e11a1239352a479053-Paper.pdf
    '''
    all_exposed_cards = state.get_all_exposed_cards(player_cards_to_see)
    player_card_amounts = [len(hand) for hand in state.player_hands]
    all_possible_hands = generate_all_possible_hands(_NUM_PLAYERS - len(player_cards_to_see), player_card_amounts, _DECK - all_exposed_cards)
    # if perspective_player == opp_player: return state.player_hands[perspective_player]
    #otherwise, the other players can have all possible combinations of unexposed cards from the deck remaining
    valid_hand_combinations = []
    valid = True
    for hands in all_possible_hands:
        played_so_far: list[set[Card]] = [set() for _ in range(_NUM_PLAYERS)]
        for round in state.previous_tricks:
            for player in range(state.who_started_tricks[round], state.who_started_tricks[round] + _NUM_PLAYERS):
                curr = player % _NUM_PLAYERS
                if not is_valid_move(state.previous_tricks[round][0], state.previous_tricks[round][curr], hands[curr] - played_so_far[curr]): valid = False
                played_so_far[curr].add(state.previous_tricks[round][curr])
        if valid: 
            for player in sorted(player_cards_to_see): hands.insert(player, state.player_hands[player])
            valid_hand_combinations.append(hands)
    return valid_hand_combinations

def get_all_histories_and_reach_probabilities(game, policies: list[Policy], player: int, subgame_root_state: WizardState, 
    player_cards_to_see: set = set(), #if we want to deal cards to a player that don't overlap with other player's cards
    players_to_exclude_in_reach_probability: set = set()):
    
    all_hands = get_all_hands_consistent_with_observations(subgame_root_state, player_cards_to_see)
    reach_probs = []
    sum_of_reach_probs = 0
    for hands in all_hands:
        game_root: WizardState = game.new_initial_state()
        joint_prob = 1
        game_root.player_hands = hands
        game_root.trump_card = subgame_root_state.trump_card
        
        for prediction in subgame_root_state.predictions:
            p = policies[game_root._next_player].action_probabilities(game_root, game_root._next_player)[prediction]
            if player not in players_to_exclude_in_reach_probability: joint_prob *= p
            game_root._apply_action(prediction)
        
        #now, we can ignore chance nodes and just find the probability of reaching this state
        #just with the given hands
        for previous_trick in game_root.previous_tricks:
            for card in previous_trick:
                action = card_to_action(card)
                p = policies[game_root._next_player].action_probabilities(game_root, game_root._next_player)[action]
                if player not in players_to_exclude_in_reach_probability: joint_prob *= p
                game_root._apply_action(action)

        for card in game_root.current_round_cards:
            action = card_to_action(card)
            p = policies[game_root._next_player].action_probabilities(game_root, game_root._next_player)[action]
            if player not in players_to_exclude_in_reach_probability: joint_prob *= p
            game_root.apply_action(action)
        sum_of_reach_probs += joint_prob
        reach_probs.append((len(reach_probs), joint_prob))
    return all_hands, reach_probs, sum_of_reach_probs

#It appears that open spiel best response maximizes CBR so i will assume this
def create_unsafe_subgame(policies: list[Policy], player: int, subgame_root_state : WizardState) -> str:
    '''Intended use: train CFR for a bit, pass in the average policy here and the root
        state from which we start subgame solving, then this will do unsafe subgame solving
        as described in https://www.cs.cmu.edu/~sandholm/safeAndNested.aaa17WS.pdf
        "player" is the player for which we want to compute the strategy

        This registers the game with openspiel and returns the name of the registered game
        
        This only allows for subgame solving beginning after the prediction stage of the round (this is solely
        because we assume the only chance action in the subgame is picking which history to go to, this could be changed
        pretty easily but doesn't seem necessary at the time of writing this).
        
        TODO this still isn't tested but it at least will run and do something it seems
    '''
    assert _NUM_PLAYERS == 2 #we can probably remove this assumption later, but for now it's needed

    game = pyspiel.load_game('python_wizard')

    all_hands, reach_probs, sum_of_reach_probs = get_all_histories_and_reach_probabilities(game, policies, player, subgame_root_state)
    class WizardSubgame(pyspiel.Game):
        def __init__(self, params=None):
            super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())
        def new_initial_state(self):
            return WizardSubgameState(self)
        def make_py_observer(self, iig_obs_type=None, params=None):
            return WizardObserver(
                iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True), #TODO this was False before...
            {'metadata': 2})
    
    class WizardSubgameState(WizardState):
        '''This is just a wrapper around the WizardSubgameState since it's the same thing
        '''
        def __init__(self, game):
            super().__init__(game)
            self.initialized = False
            self.metadata = 1
        
        def chance_outcomes(self):
            #note that dict.values() and dict.keys() return items in the same order so things don't get shuffled
            return reach_probs
        
        def is_chance_node(self):
            return not self.initialized

        def _apply_action(self, action):
            if self.is_chance_node():
                self.previous_tricks = copy(subgame_root_state.previous_tricks)
                self.current_round_cards = copy(subgame_root_state.current_round_cards)
                self.predictions = copy(subgame_root_state.predictions)
                self.tricks_per_player = copy(subgame_root_state.tricks_per_player)
                self.trump_card = subgame_root_state.trump_card
                self.current_lead_suit = subgame_root_state.current_lead_suit
                self._next_player = subgame_root_state._next_player
                #python hack to get ith element of all_hands iterators
                self.player_hands = next((x for i, x in enumerate(all_hands) if i == action), None)
                self.initialized = True
                self.metadata = 0
            else:
                assert not super().is_chance_node()
                super()._apply_action(action)
        
        def returns(self):
            #because we normalize the probability of reaching at the start, for consistentency we need to re-normalize
            return [sum_of_reach_probs * payout for payout in super().returns()]

    #now, reach_probs is a dictionary from product of hands to joint_prob; so, we can make our game now
    name = f'unsafe_wizard_solve_{str(subgame_root_state)}_player_{player}'
    _GAME_TYPE = pyspiel.GameType(
        short_name=name,
        long_name=name,
        dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
        chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
        information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
        utility=pyspiel.GameType.Utility.ZERO_SUM, #TODO this is a lie
        reward_model=pyspiel.GameType.RewardModel.TERMINAL,
        max_num_players=_NUM_PLAYERS,
        min_num_players=_NUM_PLAYERS,
        provides_information_state_string=True,
        provides_information_state_tensor=True,
        provides_observation_string=True,
        provides_observation_tensor=True,
        provides_factored_observation_string=True)
    _GAME_INFO = pyspiel.GameInfo(
        num_distinct_actions=_NUM_CARDS_PER_PLAYER+1+len(_DECK), #either a bet action (_NUM_CARDS_PER_PLAYER+1) or playing a card
        max_chance_outcomes=len(reach_probs), #TODO this overflows if it's too big
        num_players=_NUM_PLAYERS,
        min_utility=-_NUM_CARDS_PER_PLAYER*10.0,
        max_utility=20.0+10*_NUM_CARDS_PER_PLAYER,
        utility_sum=0,
        max_game_length=len(_DECK))  #from the start of the subgame TODO this isn't tight

    # best_response.joint_action_probabilities_counterfactual(root_state)
    pyspiel.register_game(_GAME_TYPE, WizardSubgame)
    return name

#for re-solve subgame solving, we assume access to a black box function that gives us
#player i's estimated expected utility at an info state (presumably from a coarse abstraction)
def blueprint_cbv_at_state(state: WizardState, player: int, blueprint_strategy: Policy) -> float: pass

def resolve_subgame_solve(cfr_policy: Policy, player: int, subgame_root_state: WizardState, blueprint_strategy: Policy) -> str:
    #just as in unsafe subgame solving, we need to enumerate all histories, however, now we connect to each history
    #in proportion to just the opponents probability of reaching it (independent of ours)
    assert _NUM_PLAYERS == 2 #we can probably remove this assumption later, but for now it's needed
    assert all([len(x) == len(subgame_root_state.player_hands[0]) for x in subgame_root_state.player_hands]) #can also remove later but makes generating all possible hands easier

    game = pyspiel.load_game('python_wizard')

    #TODO same concern with this line as above
    best_response_policy: Policy = BestResponsePolicy(game, 1-player, cfr_policy)
    policies = [cfr_policy, best_response_policy] if player == 0 else [best_response_policy, cfr_policy]

    all_hands, reach_probs, sum_of_reach_probs = get_all_histories_and_reach_probabilities(game, policies, player, subgame_root_state, set([1-player]))
    class WizardSubgame(pyspiel.Game):
        def __init__(self, params=None):
            super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())
        def new_initial_state(self):
            return WizardSubgameState(self)
        def make_py_observer(self, iig_obs_type=None, params=None):
            return WizardObserver(
                iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True), #TODO this was False before...
            {'metadata': 2})

    class WizardSubgameState(WizardState):
        '''This is just a wrapper around the WizardSubgameState since it's the same thing
        '''
        def __init__(self, game):
            super().__init__(game)
            self.initialized = False
            self.played_into_subgame: bool = False
            self.subgame_payout: float | None = None
            self.metadata = 2

        def chance_outcomes(self):
            #note that dict.values() and dict.keys() return items in the same order so things don't get shuffled
            return reach_probs
        
        def is_chance_node(self):
            return not self.initialized

        def _legal_actions(self, player) -> list:
            assert player >= 0
            if not self.played_into_subgame:
                return [0,1]
            return super()._legal_actions(player)
        
        def _apply_action(self, action):
            if self.is_chance_node():
                self.previous_tricks = copy(subgame_root_state.previous_tricks)
                self.current_round_cards = copy(subgame_root_state.current_round_cards)
                self.predictions = copy(subgame_root_state.predictions)
                self.tricks_per_player = copy(subgame_root_state.tricks_per_player)
                self.trump_card = subgame_root_state.trump_card
                self.current_lead_suit = subgame_root_state.current_lead_suit
                self._next_player = subgame_root_state._next_player
                #python hack to get ith element of all_hands iterators
                self.player_hands = next((x for i, x in enumerate(all_hands) if i == action), None)
                self.initialized = True
                self.metadata = 1
            elif self.metadata == 1:
                assert not super().is_chance_node()
                if action == 0:
                    self._game_over = True
                    self.subgame_payout = blueprint_cbv_at_state(self, player, blueprint_strategy)
                else:
                    self.metadata = 0
            else:
                super()._apply_action(action)
        
        def returns(self):
            if self.subgame_payout is not None: res = [self.subgame_payout, -self.subgame_payout] if player == 0 else [-self.subgame_payout, self.subgame_payout]
            else: res = super().returns()
            return [payout * sum_of_reach_probs for payout in res]

    #now, reach_probs is a dictionary from product of hands to joint_prob; so, we can make our game now
    name = f'resolve_wizard_solve_{str(subgame_root_state)}_player_{player}'
    _GAME_TYPE = pyspiel.GameType(
        short_name=name,
        long_name=name,
        dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
        chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
        information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
        utility=pyspiel.GameType.Utility.ZERO_SUM, #TODO this is a lie
        reward_model=pyspiel.GameType.RewardModel.TERMINAL,
        max_num_players=_NUM_PLAYERS,
        min_num_players=_NUM_PLAYERS,
        provides_information_state_string=True,
        provides_information_state_tensor=True,
        provides_observation_string=True,
        provides_observation_tensor=True,
        provides_factored_observation_string=True)
    _GAME_INFO = pyspiel.GameInfo(
        #this is kinda hacky but we want to interface with the normal WizardState so even though subgames
        #have no betting, we still pretend they do, and use that range of actions to instead choose to subgame or not subgame
        num_distinct_actions=_NUM_CARDS_PER_PLAYER+1+len(_DECK), #either playing a card or entering subgame or not
        max_chance_outcomes=len(reach_probs), #TODO this overflows if it's too big
        num_players=_NUM_PLAYERS,
        min_utility=-_NUM_CARDS_PER_PLAYER*10.0,
        max_utility=20.0+10*_NUM_CARDS_PER_PLAYER,
        utility_sum=0,
        max_game_length=len(_DECK))  #from the start of the subgame TODO this isn't tight

    # best_response.joint_action_probabilities_counterfactual(root_state)
    pyspiel.register_game(_GAME_TYPE, WizardSubgame)
    return name

def maxmargin_subgame_solve(cfr_policy: Policy, player: int, subgame_root_state: WizardState, blueprint_strategy: Policy) -> str:
    '''See https://ojs.aaai.org/index.php/AAAI/article/view/10033/9892 (specifically the gadget game / extensive form version of this)
    Essentially, we offset all histories to have the same cbv and allow our opponent to choose which infoset they enter (ie., what their hand is)
    since they will always choose the one with the minimum margin), and this choice connects to a chance node that randomizes over the possible hands
    of the other players to have gotten to that point and that makes it so that our goal is to maximize the minimum margin.

    TODO we could probably (without much trouble) get subgame solving working against multiple players by just taking in
    blueprint strategies for each of the other players
    '''
    assert _NUM_PLAYERS == 2 #we can probably remove this assumption later, but for now it's needed

    game = pyspiel.load_game('python_wizard')

    #TODO same concern with this line as above
    best_response_policy: Policy = BestResponsePolicy(game, 1-player, cfr_policy)
    policies = [cfr_policy, best_response_policy] if player == 0 else [best_response_policy, cfr_policy]

    class WizardSubgame(pyspiel.Game):
        def __init__(self, params=None):
            super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())
        def new_initial_state(self):
            return WizardSubgameState(self)
        def make_py_observer(self, iig_obs_type=None, params=None):
            return WizardObserver(
                iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True), #TODO this was False before...
            {'metadata': 2})

    class WizardSubgameState(WizardState):
        '''This is just a wrapper around the WizardSubgameState since it's the same thing
        '''
        def __init__(self, game):
            super().__init__(game)
            self.initialized = False
            self.subgame_payout: float | None = None 
            self.all_consistent_hands: list = get_all_hands_consistent_with_observations(self)
            self.metadata = 2

        def chance_outcomes(self):
            #in this scenario, we need to compute the probability of reaching this state with the opposing player's hand fixed and assume they try to reach it
            all_hands, reach_probs, sum_of_reach_probs = get_all_histories_and_reach_probabilities(game, policies, player, subgame_root_state, 
                    players_to_exclude_in_reach_probability=set([1-player]), player_cards_to_see=set([1-player]))
            self.all_consistent_hands = all_hands #set of all hands consistent with the game history
            return reach_probs
        
        def is_chance_node(self):
            return self.subgame_payout is not None and not self.initialized

        def _legal_actions(self, player) -> list:
            assert player >= 0
            if self.subgame_payout is None:
                #the opponent player can pick whatever hand they want consistent with their observations
                return range(len(get_all_hands_consistent_with_observations(self)))
            return super()._legal_actions(player)
        
        def _apply_action(self, action):
            if self.is_chance_node():
                self.previous_tricks = copy(subgame_root_state.previous_tricks)
                self.current_round_cards = copy(subgame_root_state.current_round_cards)
                self.predictions = copy(subgame_root_state.predictions)
                self.tricks_per_player = copy(subgame_root_state.tricks_per_player)
                self.trump_card = subgame_root_state.trump_card
                self.current_lead_suit = subgame_root_state.current_lead_suit
                self._next_player = subgame_root_state._next_player
                #python hack to get ith element of all_hands iterators
                self.player_hands[player] = list(combinations())
                self.initialized = True
            elif self.subgame_payout is None:
                #let our opponent pick a hand from among their possible hands they can have
                self.player_hands[1-player] = self.all_consistent_hands[action][player]
                self.subgame_payout = blueprint_cbv_at_state(self, player, blueprint_strategy)
                self.metadata = 0
            else:
                super()._apply_action(action)
        def returns(self):
            '''According to the paper, the game payouts are normal returns but we subtract the cbv
                from the start of each infoset in the subgame (which in Wizard, augmented infosets = infosets) 
            '''
            normal_returns = super().returns()
            assert _NUM_PLAYERS == 2
            if player == 0: return [normal_returns[0] - self.subgame_payout, normal_returns[1] + self.subgame_payout]
            else: return [normal_returns[0] + self.subgame_payout, normal_returns[1] - self.subgame_payout]

    #now, reach_probs is a dictionary from product of hands to joint_prob; so, we can make our game now
    name = f'resolve_wizard_solve_{str(subgame_root_state)}_player_{player}'
    _GAME_TYPE = pyspiel.GameType(
        short_name=name,
        long_name=name,
        dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
        chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
        information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
        utility=pyspiel.GameType.Utility.ZERO_SUM, #TODO this is a lie
        reward_model=pyspiel.GameType.RewardModel.TERMINAL,
        max_num_players=_NUM_PLAYERS,
        min_num_players=_NUM_PLAYERS,
        provides_information_state_string=True,
        provides_information_state_tensor=True,
        provides_observation_string=True,
        provides_observation_tensor=True,
        provides_factored_observation_string=True)
    _GAME_INFO = pyspiel.GameInfo(
        #normal actions (although there will be no prediction stage) + the ability for the opposing player to pick their hand
        num_distinct_actions=_NUM_CARDS_PER_PLAYER+1+len(_DECK)+math.comb(len(_DECK), len(root_state.player_hands[1 - player])), #either playing a card or entering subgame or not
        max_chance_outcomes=2**64, #just put some upper bound on this
        num_players=_NUM_PLAYERS,
        min_utility=-_NUM_CARDS_PER_PLAYER*10.0,
        max_utility=20.0+10*_NUM_CARDS_PER_PLAYER,
        utility_sum=0,
        max_game_length=len(_DECK))  #from the start of the subgame TODO this isn't tight

    # best_response.joint_action_probabilities_counterfactual(root_state)
    pyspiel.register_game(_GAME_TYPE, WizardSubgame)
    return name


if __name__ == '__main__':
    game = pyspiel.load_game('python_wizard')
    policies = [UniformRandomPolicy(game), UniformRandomPolicy(game)]
    root_state: WizardState = game.new_initial_state()
    root_state.player_hands = [set([Card(Face.ACE, Suit.DIAMOND), Card(Face.WIZARD, Suit.CLUB)]), set([Card(Face.WIZARD, Suit.DIAMOND), Card(Face.JACK, Suit.DIAMOND)])]
    root_state.trump_card = Card(Face.KING, Suit.DIAMOND)
    root_state.who_started_tricks.append(0)
    root_state.predictions = [1,1]
    unsafe_subgame = create_unsafe_subgame(policies, 0, root_state)
    subgame = pyspiel.load_game(unsafe_subgame)
    print(exploitability.exploitability(game, policies[0]))
    #now, run 1 iteration of CFR on the subgame
    cfr_solver = cfr.CFRSolver(subgame)
    for i in range(10):
        print(f'Running CFR iteration {i}')
        cfr_solver.evaluate_and_update_policy()
    write_subgame_strategy_onto_policy_for_player(policies[0].to_tabular(), unsafe_subgame, cfr_solver.average_policy(), 0)
    print(exploitability.exploitability(game, policies[0]))
    