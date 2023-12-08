from absl import app
from absl import flags

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
import pyspiel
import pickle
import wizard

FLAGS = flags.FLAGS

# flags.DEFINE_integer("iterations", 100, "Number of iterations")
# flags.DEFINE_string("game", "wizard_python", "Name of the game")
# flags.DEFINE_integer("players", 3, "Number of players")
# flags.DEFINE_integer("print_freq", 1, "How often to print the exploitability")

def main():
  game = pyspiel.load_game('python_wizard')
  cfr_solver = cfr.CFRSolver(game)

  for i in range(10):
    cfr_solver.evaluate_and_update_policy()
    # if i % 1 == 0:
    #   conv = exploitability.exploitability(game, cfr_solver.average_policy())
    #   print("Iteration {} exploitability {}".format(i, conv))
    with open(filename, "wb") as file:
      pickle.dump(solver, file, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()