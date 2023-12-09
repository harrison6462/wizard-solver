## Wizard Solver

# Overview
In this repository is our code to solve Wizard, an imperfect information trick taking game. This is a project for Carnegie Mellon's 15-888 Computational Game Solving for Fall 2023 https://www.cs.cmu.edu/~sandholm/cs15-888F23/. This was all done in the OpenSpiel library from DeepMind.

For a high level overview of where code for things is located:

wizard.py contains the core logic of the game.

abstraction.py computes the card abstractions as specified in our paper.

dummy_mcts.py contains the code for IS-MCTS as described in the paper.

subgame_solve_wizard.py contains implementations for unsafe endgame solving, re-solve refinement, and maxmargin refinement as specified in the paper.

abstracted_wizard.py contains the code for computing the abstraction and corresponding between the abstracted / base game.

Then, there are a few testing scripts, whose names are self explanatory. To actually play against the computed policies, use play_against_bot.py, and see the parameters in there for use.