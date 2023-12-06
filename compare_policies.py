from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
import pyspiel
import wizard
import time
import os
import pickle

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()

game = pyspiel.load_game("python_wizard")
print("loaded game\n")

# change these two lines to choose policies
# policy2 = "avg_policy100(layers64).pickle" 
policy2 = "avg_policy100.pickle" 
policy1 = "avg_policy100(layers64).pickle"

with open(policy1, "rb") as file:
    loaded_policy1 = pickle.load(file)

with open(policy2, "rb") as file:
    loaded_policy2 = pickle.load(file)
print("loaded policies\n")

p1_against_p2 = expected_game_score.policy_value(
    game.new_initial_state(), [loaded_policy1, loaded_policy2])
print(f"Computed {policy1} value against {policy2}: {p1_against_p2}")

# conv = exploitability.nash_conv(game, loaded_policy1)
# print(f"polcy1 - NashConv: {conv}")

# conv = exploitability.nash_conv(game, loaded_policy2)
# print(f"policy2 - NashConv: {conv}")
