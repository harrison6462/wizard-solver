# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example use of the CFR algorithm on Kuhn Poker."""

import pickle
import sys
from absl import app
from absl import flags

import pyspiel

import time

import wizard

import new_wizard_abstracted

FLAGS = flags.FLAGS

flags.DEFINE_enum("solver", "cfrplus", ["cfr", "cfrplus", "cfrbr"], "CFR solver")
flags.DEFINE_integer("iterations", 80, "Number of iterations")
flags.DEFINE_string("game", "python_wizard", "Name of the game")

def main(_):
  game = pyspiel.load_game(
      FLAGS.game,
  )
  print("loaded game\n")

  s = time.time()
  solver = None
  if FLAGS.solver == "cfr":
    solver = pyspiel.CFRSolver(game)
  elif FLAGS.solver == "cfrplus":
    solver = pyspiel.CFRPlusSolver(game)
  elif FLAGS.solver == "cfrbr":
    solver = pyspiel.CFRBRSolver(game)
  else:
    print("Unknown solver")
    sys.exit(0)
  filename = "cfrplus_14card_abstract.pickle".format(FLAGS.solver)
  # with open(filename, 'rb') as f:
  #   solver = pickle.load(f)
  e = time.time()
  print(f"created solver in {e-s} seconds\n")

  for i in range(int(FLAGS.iterations)):
    solver.evaluate_and_update_policy()
    print(f'Ran iteration{i}')
    # if i % 10 == 0:
    #   print("Iteration {} exploitability: {:.6f}".format(
    #       i, pyspiel.exploitability(game, solver.average_policy())))

  # print("Persisting the model...")
    if i % 3 == 0:
      with open(filename, "wb") as file:
        pickle.dump(solver, file, pickle.HIGHEST_PROTOCOL)

  # print("Loading the model...")
  # with open(filename, "rb") as file:
  #   loaded_solver = pickle.load(file)
  # print("Exploitability of the loaded model: {:.6f}".format(
  #     pyspiel.exploitability(game, loaded_solver.average_policy())))

  # for i in range(int(FLAGS.iterations / 2)):
  #   loaded_solver.evaluate_and_update_policy()
  #   tabular_policy = loaded_solver.tabular_average_policy()
  #   print(f"Tabular policy length: {len(tabular_policy)}")
  #   print("Iteration {} exploitability: {:.6f}".format(
  #       int(FLAGS.iterations / 2) + i,
  #       pyspiel.exploitability(game, loaded_solver.average_policy())))
  print(pyspiel.exploitability(game, solver.average_policy()))
  with open(filename, "wb") as file:
    pickle.dump(solver, file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
  # print(f'standard cfr started at {time.time() - start}\n')
  app.run(main)
  # print(f'standard cfr completed in {time.time() - start} seconds\n')
