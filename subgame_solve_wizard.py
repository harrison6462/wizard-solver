from __future__ import annotations

from copy import copy, deepcopy
from enum import Enum
import wizard_cfr_cpp
from wizard import Deck, Card, Hand, Face, Suit, _NUM_PLAYERS, _NUM_CARDS_PER_PLAYER, _DECK, WizardGame, WizardObserver, WizardState, card_to_action, generate_all_possible_hands, is_valid_move
from open_spiel.python.algorithms.best_response import BestResponsePolicy
from open_spiel.python.policy import UniformRandomPolicy, Policy, TabularPolicy
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
from itertools import product, combinations
import math
import pickle
import pyspiel

def write_subgame_strategy_onto_policy_for_player(blueprint_strategy: TabularPolicy, subgame_name: str, subgame_policy: TabularPolicy, players_to_update: set) -> None:
    '''This will be kind of hacky, but not too different from how open spiel would handle this normally.
        We have the two tabular subgames
    '''
    subgame: WizardGame = pyspiel.load_game(subgame_name)
    observer = subgame.make_py_observer()

    for state in subgame_policy.states:
        if state.is_terminal(): continue
        if state._next_player not in players_to_update or state.metadata != 0: continue
        try:
            state_policy = blueprint_strategy.action_probabilities(observer.string_from(state, state._next_player))
            for action, value in subgame_policy.action_probabilities(state, state._next_player).items():
                state_policy[action] = value
        except LookupError as e:
            print('Failed to find key ', e)

def get_all_hands_consistent_with_observations(state: WizardState, player_cards_to_see: set = set()) -> list[list[Hand]]:
    '''Returns all possible hands the player can have from deck consistent with
    the observations (ie., if the lead suit is clubs and they play a non-wizard/jester
    non-club then they cannot have clubs).
    This is to be able to compute all possible histories as required by subgame solving.
    In fancier words, this is returning the common knowledge closure of the given state,
    as defined in https://proceedings.neurips.cc/paper/2021/file/c96c08f8bb7960e11a1239352a479053-Paper.pdf
    '''
    all_exposed_cards = state.get_all_exposed_cards(player_cards_to_see)
    unseen_card_amounts = [len(state.player_hands[i]) for i in range(len(state.player_hands)) if i not in player_cards_to_see]
    if len(unseen_card_amounts) == 0: return [deepcopy(state.player_hands)]
    all_possible_hands = generate_all_possible_hands(_NUM_PLAYERS - len(player_cards_to_see), unseen_card_amounts, _DECK - all_exposed_cards)
    # if perspective_player == opp_player: return state.player_hands[perspective_player]
    #otherwise, the other players can have all possible combinations of unexposed cards from the deck remaining
    valid_hand_combinations = []
    
    for hands in all_possible_hands:
        valid = True

        for player in sorted(player_cards_to_see): hands.insert(player, copy(state.player_hands[player]))
        played_so_far: list[set[Card]] = [set() for _ in range(_NUM_PLAYERS)]
        for round in state.previous_tricks:
            for player in range(state.who_started_tricks[round], state.who_started_tricks[round] + _NUM_PLAYERS):
                curr = player % _NUM_PLAYERS
                if not is_valid_move(state.previous_tricks[round][0], state.previous_tricks[round][curr], hands[curr] - played_so_far[curr]): 
                    valid = False
                    break
                played_so_far[curr].add(state.previous_tricks[round][curr])
        if valid: 
            valid_hand_combinations.append(hands)
    return valid_hand_combinations

def get_all_initial_hands_consistent_with_state(state: WizardState, player_cards_to_see: set = set()) -> list[list[Hand]]:
    all_hands = get_all_hands_consistent_with_observations(state, player_cards_to_see)
    played_cards: list[set[Card]] = [set() for _ in range(_NUM_PLAYERS)]
    if len(state.who_started_tricks) > 0:
        for i in range(len(state.previous_tricks)):
            for trick in range(len(state.previous_tricks[i])):
                for player in range(state.who_started_tricks[i], state.who_started_tricks[i] + _NUM_PLAYERS):
                    played_cards[player % _NUM_PLAYERS].add(trick[player % _NUM_PLAYERS])
        
        for player in range(state.who_started_tricks[-1], state.who_started_tricks[-1] + len(state.current_round_cards)):
            played_cards[player % _NUM_PLAYERS].add(state.current_round_cards[player % _NUM_PLAYERS])

        for hands in all_hands:
            for player in range(_NUM_PLAYERS): 
                hands[player] = hands[player].union(played_cards[player])
    return all_hands

def get_history_reach_probability(game, policies: list[Policy], subgame_root_state: WizardState, players_to_exclude_in_reach_probability: set):
    game_root: WizardState = game.new_initial_state()
    joint_prob = 1
    game_root.player_hands = copy(subgame_root_state.initial_player_hands)
    game_root.trump_card = subgame_root_state.trump_card
    game_root.initial_player_hands = deepcopy(subgame_root_state.initial_player_hands)
    assert len(subgame_root_state.predictions) == _NUM_PLAYERS
    try:
        for prediction in subgame_root_state.predictions:
            if game_root._next_player not in players_to_exclude_in_reach_probability: 
                p = policies[game_root._next_player].action_probabilities(game_root)[prediction]
                joint_prob *= p
            game_root._apply_action(prediction)
        
        #now, we can ignore chance nodes and just find the probability of reaching this state
        #just with the given hands
        for previous_trick in subgame_root_state.previous_tricks:
            for card in previous_trick:
                action = card_to_action(card)
                if game_root._next_player not in players_to_exclude_in_reach_probability: 
                    p = policies[game_root._next_player].action_probabilities(game_root)[action]
                    joint_prob *= p
                game_root._apply_action(action)

        for card in subgame_root_state.current_round_cards:
            action = card_to_action(card)
            if game_root._next_player not in players_to_exclude_in_reach_probability: 
                p = policies[game_root._next_player].action_probabilities(game_root, game_root._next_player)[action]
                joint_prob *= p
            game_root._apply_action(action)
    except KeyError as e:
        if isinstance(policies[game_root._next_player], BestResponsePolicy):
            #if we fail to find a policy in a best response policy, assume it's because it's not the best action
            return 0
        #otherwise, our tabular policy is missing a state - that's bad!
        raise e
    return joint_prob

def get_all_histories_and_joint_reach_probabilities(game, policies_per_player: list[Policy], subgame_root_state: WizardState, 
    player_cards_to_see: set = set(), #if we want to deal cards to a player that don't overlap with other player's cards
    players_to_exclude_in_reach_probability: set = set()):
    
    all_possible_initial_hands = get_all_initial_hands_consistent_with_state(subgame_root_state, player_cards_to_see)
    reach_probs = []
    sum_of_reach_probs = 0
    dummy_subgame = deepcopy(subgame_root_state)
    for hands in all_possible_initial_hands:
        dummy_subgame.player_hands = hands
        dummy_subgame.initial_player_hands = deepcopy(hands)
        joint_prob = get_history_reach_probability(game, policies_per_player, dummy_subgame, players_to_exclude_in_reach_probability)
        sum_of_reach_probs += joint_prob
        reach_probs.append((len(reach_probs), joint_prob))
    if sum_of_reach_probs == 0: raise Exception('Reaching state was impossible', subgame_root_state)
    return all_possible_initial_hands, [(i, p/sum_of_reach_probs) for i, p in reach_probs], sum_of_reach_probs

#for re-solve subgame solving, we assume access to a black box function that gives us
#player i's estimated expected utility at an info state (presumably from a coarse abstraction)
def blueprint_cbv_at_state(game: WizardGame, state: WizardState, player: int, blueprint_strategy: Policy, blueprint_best_response: BestResponsePolicy) -> float: 
    '''I'm surprised that openspiel doesn't seem to have a function that computes the counterfactual best response at a given state,
        but here we are. The counterfactual best response value is the expected utility of player "player" given that player p
        plays to reach the state, and then employs a counterfactual best response against the opponents strategy (the blueprint).
        Formally, it's = (sum over all histories in this infoset of the probability our opponents played this history * ...)

        TODO It appears that open spiel best response is also a CBR (since it calculates the best response at every infoset and not just the reachable
        ones), so I will assume this to be the case
    '''
    assert _NUM_PLAYERS == 2
    #to compute the cbv, we (unfortunately) have to do a tree search of all rollouts and seeing who wins them
    all_opponent_hands, history_reach_probs, sum_of_reach_probs = get_all_histories_and_joint_reach_probabilities(game, [blueprint_strategy for _ in range(_NUM_PLAYERS)], state, 
                                                                                                        player_cards_to_see=set([player]), players_to_exclude_in_reach_probability=set([player]))
    #set the game to be in normal wizard playing mode
    original_metadata = state.metadata
    state.metadata = 0
    acc = 0
    for hand in all_opponent_hands:
        #this is gonna be insanely inefficient but lets DFS this
        init = deepcopy(state)
        init.player_hands = hand
        init.initial_player_hands = deepcopy(hand)
        stack: list[tuple[WizardState, float]] = [(init, 1)]
        val = 0
        while len(stack) > 0:
            curr_state, reach_prob = stack.pop()
            if curr_state.is_terminal(): 
                val += curr_state.returns()[player] * reach_prob

            #TODO we could easily implement this in general but this shouldn't happen in Wizard
            elif curr_state.is_chance_node(): raise Exception("Wizard shouldn't have any chance nodes in the subgame tree.")
            else:
                for action in curr_state._legal_actions(curr_state._next_player):
                    try:
                        if curr_state._next_player == player: 
                            new_reach_prob = reach_prob * blueprint_best_response.action_probabilities(curr_state, player)[action]
                        else: 
                            new_reach_prob = reach_prob *  blueprint_strategy.action_probabilities(curr_state, curr_state._next_player)[action]
                    except KeyError as e: #open spiel lets you not insert 0 probability actions...
                        # print(f"Strategy didn't have action {e} at infoset {str(curr_state)}")
                        new_reach_prob = 0
                    if new_reach_prob == 0: continue
                    new_state = deepcopy(curr_state)
                    new_state._apply_action(action)
                    stack.append((new_state, new_reach_prob))
                #since we ignore the probabilities for player "player", we can pretend it's just the same policy as the other players
        acc += val * get_history_reach_probability(game, [blueprint_strategy for _ in range(_NUM_PLAYERS)], init, players_to_exclude_in_reach_probability=set([player]))
    state.metadata = original_metadata
    return acc/sum_of_reach_probs

def create_unsafe_subgame(policy: Policy, subgame_root_state : WizardState) -> str:
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

    all_hands, reach_probs, sum_of_reach_probs = get_all_histories_and_joint_reach_probabilities(game, [policy for _ in range(_NUM_PLAYERS)], subgame_root_state)
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
            self.metadata = 1
            self.previous_tricks = copy(subgame_root_state.previous_tricks)
            self.current_round_cards = copy(subgame_root_state.current_round_cards)
            self.predictions = copy(subgame_root_state.predictions)
            self.tricks_per_player = copy(subgame_root_state.tricks_per_player)
            self.who_started_tricks = copy(subgame_root_state.who_started_tricks)
            self.trump_card = subgame_root_state.trump_card
            self.current_lead_suit = subgame_root_state.current_lead_suit
            self._next_player = subgame_root_state._next_player
        
        def chance_outcomes(self):
            #note that dict.values() and dict.keys() return items in the same order so things don't get shuffled
            return reach_probs
        
        def is_chance_node(self):
            return self.metadata == 1

        def _apply_action(self, action):
            if self.is_chance_node():
                #python hack to get ith element of all_hands iterators
                self.player_hands = next((x for i, x in enumerate(all_hands) if i == action), None)
                self.metadata = 0
            else:
                assert not super().is_chance_node()
                super()._apply_action(action)
        
        def returns(self):
            #because we normalize the probability of reaching at the start, for consistentency we need to re-normalize
            return [sum_of_reach_probs * payout for payout in super().returns()]

    #now, reach_probs is a dictionary from product of hands to joint_prob; so, we can make our game now
    name = f'unsafe_wizard_solve_{str(subgame_root_state)}'
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
        min_utility=-2*_NUM_CARDS_PER_PLAYER*10.0, #TODO this is wrong
        max_utility=20.0+10*_NUM_CARDS_PER_PLAYER,
        utility_sum=0,
        max_game_length=len(_DECK))  #from the start of the subgame TODO this isn't tight

    # best_response.joint_action_probabilities_counterfactual(root_state)
    pyspiel.register_game(_GAME_TYPE, WizardSubgame)
    return name

def create_resolve_subgame(blueprint_strategy: Policy, player: int, subgame_root_state: WizardState) -> str:
    #just as in unsafe subgame solving, we need to enumerate all histories, however, now we connect to each history
    #in proportion to just the opponents probability of reaching it (independent of ours)
    assert _NUM_PLAYERS == 2 #we can probably remove this assumption later, but for now it's needed
    assert all([len(x) == len(subgame_root_state.player_hands[0]) for x in subgame_root_state.player_hands]) #can also remove later but makes generating all possible hands easier

    game = pyspiel.load_game('python_wizard')

    best_response_policy: Policy = BestResponsePolicy(game, player, blueprint_strategy)

    all_hands, reach_probs, sum_of_reach_probs = get_all_histories_and_joint_reach_probabilities(game, [blueprint_strategy for _ in range(_NUM_PLAYERS)], subgame_root_state, players_to_exclude_in_reach_probability=set([1-player]))
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
            self.subgame_payout: float | None = None
            self.metadata = 2
            self.previous_tricks = copy(subgame_root_state.previous_tricks)
            self.current_round_cards = copy(subgame_root_state.current_round_cards)
            self.predictions = copy(subgame_root_state.predictions)
            self.tricks_per_player = copy(subgame_root_state.tricks_per_player)
            self.who_started_tricks = copy(subgame_root_state.who_started_tricks)
            self.player_hands = deepcopy(subgame_root_state.player_hands)
            self.initial_player_hands = deepcopy(subgame_root_state.initial_player_hands)
            self.trump_card = subgame_root_state.trump_card
            self.current_lead_suit = subgame_root_state.current_lead_suit
            self._next_player = subgame_root_state._next_player
            
        def chance_outcomes(self):
            #note that dict.values() and dict.keys() return items in the same order so things don't get shuffled
            return reach_probs
        
        def is_chance_node(self):
            return self.metadata == 2

        def _legal_actions(self, player) -> list:
            assert player >= 0
            if self.metadata == 1: return [0,1]
            return super()._legal_actions(player)
        
        def is_terminal(self):
            return super().is_terminal() or self.subgame_payout is not None

        def _apply_action(self, action):
            if self.is_chance_node():
                #python hack to get ith element of all_hands iterators
                self.player_hands = next((x for i, x in enumerate(all_hands) if i == action), None)
                self.metadata = 1
            elif self.metadata == 1:
                assert not super().is_chance_node()
                if action == 0:
                    self.subgame_payout = blueprint_cbv_at_state(game, self, 1-player, blueprint_strategy, best_response_policy)
                else:
                    self.metadata = 0
            else:
                assert len(self.predictions) == _NUM_PLAYERS
                super()._apply_action(action)
        
        def returns(self):
            if self.subgame_payout is not None: res = [self.subgame_payout, -self.subgame_payout] if player == 0 else [-self.subgame_payout, self.subgame_payout]
            else: res = super().returns()
            return res

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
        min_utility=-2*_NUM_CARDS_PER_PLAYER*10.0,
        max_utility=20.0+10*_NUM_CARDS_PER_PLAYER,
        utility_sum=0,
        max_game_length=len(_DECK))  #from the start of the subgame TODO this isn't tight

    # best_response.joint_action_probabilities_counterfactual(root_state)
    pyspiel.register_game(_GAME_TYPE, WizardSubgame)
    return name

def create_maxmargin_subgame(blueprint_strategy: Policy, player: int, subgame_root_state: WizardState) -> str:
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
    best_response_policy: Policy = BestResponsePolicy(game, player, blueprint_strategy)
    policies = [best_response_policy, blueprint_strategy] if player == 0 else [blueprint_strategy, best_response_policy]
    class WizardSubgame(pyspiel.Game):
        def __init__(self, params=None):
            super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())
        def new_initial_state(self):
            return WizardSubgameState(self)
        def make_py_observer(self, iig_obs_type=None, params=None):
            return WizardObserver(
                iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True), #TODO this was False before...
            params)

    class WizardSubgameState(WizardState):
        '''This is just a wrapper around the WizardSubgameState since it's the same thing
        '''
        def __init__(self, game):
            super().__init__(game)
            self.subgame_payout: float | None = None 
            self.previous_tricks = copy(subgame_root_state.previous_tricks)
            self.current_round_cards = copy(subgame_root_state.current_round_cards)
            self.predictions = copy(subgame_root_state.predictions)
            self.tricks_per_player = copy(subgame_root_state.tricks_per_player)
            self.who_started_tricks = copy(subgame_root_state.who_started_tricks)
            self.player_hands = deepcopy(subgame_root_state.player_hands)
            self.initial_player_hands = deepcopy(subgame_root_state.initial_player_hands)
            self.trump_card = subgame_root_state.trump_card
            self.current_lead_suit = subgame_root_state.current_lead_suit
            self._next_player = subgame_root_state._next_player
            self.all_consistent_hands: list = get_all_hands_consistent_with_observations(self)
            self.metadata = 2

        def chance_outcomes(self):
            #in this scenario, we need to compute the probability of reaching this state with the opposing player's hand fixed and assume they try to reach it
            all_hands, reach_probs, sum_of_reach_probs = get_all_histories_and_joint_reach_probabilities(game, policies, subgame_root_state, 
                    players_to_exclude_in_reach_probability=set([player]), player_cards_to_see=set([1-player]))
            self.all_consistent_hands = all_hands #set of all hands consistent with the game history
            return reach_probs
        
        def is_chance_node(self):
            return self.metadata == 1

        def _legal_actions(self, player) -> list:
            assert player >= 0
            if self.metadata == 2:
                #the opponent player can pick whatever hand they want consistent with their observations
                return range(len(self.all_consistent_hands))
            return super()._legal_actions(player)
        
        def _apply_action(self, action):
            if self.is_chance_node():
                #NOTE self.all_consistent_hands gets updated in chance_outcomes
                self.player_hands[player] = self.all_consistent_hands[action][player]
                self.metadata = 0
            elif self.metadata == 2:
                #let our opponent pick a hand from among their possible hands they can have
                self.player_hands[1-player] = self.all_consistent_hands[action][1-player]
                self.subgame_payout = blueprint_cbv_at_state(game, self, 1-player, blueprint_strategy, best_response_policy)
                self.metadata = 1
            else:
                super()._apply_action(action)
        def returns(self):
            '''According to the paper, the game payouts are normal returns but we subtract the cbv
                from the start of each infoset in the subgame (which in Wizard, augmented infosets = infosets) 
            '''
            normal_returns = super().returns()
            assert _NUM_PLAYERS == 2
            if player == 0: return [normal_returns[0] + self.subgame_payout, normal_returns[1] - self.subgame_payout]
            else: return [normal_returns[0] - self.subgame_payout, normal_returns[1] + self.subgame_payout]

    #now, reach_probs is a dictionary from product of hands to joint_prob; so, we can make our game now
    name = f'maxmargin_wizard_solve_{str(subgame_root_state)}_player_{player}'
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
        num_distinct_actions=_NUM_CARDS_PER_PLAYER+1+len(_DECK)+math.comb(len(_DECK), _NUM_CARDS_PER_PLAYER)**2, #either playing a card or entering subgame or not
        max_chance_outcomes=math.comb(len(_DECK), _NUM_CARDS_PER_PLAYER), #just put some upper bound on this
        num_players=_NUM_PLAYERS,
        min_utility=-2*_NUM_CARDS_PER_PLAYER*10.0,
        max_utility=20.0+10*_NUM_CARDS_PER_PLAYER,
        utility_sum=0,
        max_game_length=len(_DECK)+2)  #from the start of the subgame TODO this isn't tight

    # best_response.joint_action_probabilities_counterfactual(root_state)
    pyspiel.register_game(_GAME_TYPE, WizardSubgame)
    return name

def do_unsafe_subgame_solve_test():
    game = pyspiel.load_game('python_wizard')
    policy = UniformRandomPolicy(game).to_tabular()
    with open('cfrplus_solver.pickle', 'rb') as f:
        policy = pickle.load(f).tabular_average_policy()
    for predictions in map(lambda t: [t[0], t[1]], product([0,1,2], [0,1,2])):
        root_state: WizardState = game.new_initial_state()
        root_state.player_hands = [set([Card(Face.ACE, Suit.DIAMOND), Card(Face.WIZARD, Suit.CLUB)]), set([Card(Face.WIZARD, Suit.DIAMOND), Card(Face.JESTER, Suit.CLUB)])]
        root_state.initial_player_hands = deepcopy(root_state.player_hands)
        root_state.trump_card = Card(Face.ACE, Suit.CLUB)
        root_state.predictions = predictions
        root_state._next_player = 0 if predictions[0] >= predictions[1] else 1
        root_state.who_started_tricks.append(root_state._next_player)
        unsafe_subgame = create_unsafe_subgame(policy, root_state)
        subgame = pyspiel.load_game(unsafe_subgame)
        print(exploitability.exploitability(game, policy))
        #now, run 1 iteration of CFR on the subgame
        cfr_solver = cfr.CFRSolver(subgame)
        for i in range(10):
            print(f'Running CFR iteration {i}')
            cfr_solver.evaluate_and_update_policy()
            # avg_policy = cfr_solver.average_policy()
            # print(exploitability.exploitability(subgame, cfr_solver.average_policy()))
        write_subgame_strategy_onto_policy_for_player(policy, unsafe_subgame, cfr_solver.average_policy(), set([0,1]))
    print(exploitability.exploitability(game, policy))

def do_resolve_subgame_solve_test():
    game = pyspiel.load_game('python_wizard')
    # policy = UniformRandomPolicy(game).to_tabular()
    with open('cfrplus_solver.pickle', 'rb') as f:
        policy = pickle.load(f).tabular_average_policy()
    for predictions in map(lambda t: [t[0], t[1]], product([0,1,2], [0,1,2])):
        root_state: WizardState = game.new_initial_state()
        root_state.player_hands = [set([Card(Face.ACE, Suit.DIAMOND), Card(Face.WIZARD, Suit.CLUB)]), set([Card(Face.WIZARD, Suit.DIAMOND), Card(Face.JESTER, Suit.CLUB)])]
        root_state.initial_player_hands = deepcopy(root_state.player_hands)
        root_state.trump_card = Card(Face.ACE, Suit.CLUB)
        root_state.predictions = predictions
        root_state._next_player = 0 if predictions[0] >= predictions[1] else 1
        root_state.who_started_tricks.append(root_state._next_player)
        resolve_subgame = create_resolve_subgame(policy, 0, root_state)
        subgame = pyspiel.load_game(resolve_subgame)
        # print(exploitability.exploitability(game, policy))
        cfr_solver = cfr.CFRSolver(subgame)
        for i in range(10):
            print(f'Running CFR iteration {i}')
            cfr_solver.evaluate_and_update_policy()
            avg_policy = cfr_solver.average_policy()
            # print(exploitability.exploitability(subgame, cfr_solver.average_policy()))
        write_subgame_strategy_onto_policy_for_player(policy, resolve_subgame, cfr_solver.average_policy(), set([0,1]))
    print(exploitability.exploitability(game, policy))

def do_maxmargin_subgame_solve_test():
    game = pyspiel.load_game('python_wizard')
    policy = UniformRandomPolicy(game).to_tabular()
    for predictions in map(lambda t: [t[0], t[1]], product([0,1,2], [0,1,2])):
        root_state: WizardState = game.new_initial_state()
        root_state.player_hands = [set([Card(Face.ACE, Suit.DIAMOND), Card(Face.WIZARD, Suit.CLUB)]), set([Card(Face.WIZARD, Suit.DIAMOND), Card(Face.JESTER, Suit.CLUB)])]
        root_state.initial_player_hands = deepcopy(root_state.player_hands)
        root_state.trump_card = Card(Face.ACE, Suit.CLUB)
        root_state.predictions = predictions
        root_state._next_player = 0 if predictions[0] >= predictions[1] else 1
        root_state.who_started_tricks.append(root_state._next_player)
        maxmargin_subgame = create_maxmargin_subgame(policy, 0, root_state)
        subgame = pyspiel.load_game(maxmargin_subgame)
        # print(exploitability.exploitability(game, policy))
        cfr_solver = cfr.CFRSolver(subgame)
        for i in range(10):
            print(f'Running CFR iteration {i}')
            cfr_solver.evaluate_and_update_policy()
            avg_policy = cfr_solver.average_policy()
            # print(exploitability.exploitability(subgame, cfr_solver.average_policy()))
        write_subgame_strategy_onto_policy_for_player(policy, maxmargin_subgame, cfr_solver.average_policy(), set([0,1]))
    print(exploitability.exploitability(game, policy))


if __name__ == '__main__':
    # do_unsafe_subgame_solve_test()
    do_resolve_subgame_solve_test()
    # do_maxmargin_subgame_solve_test()