import copy
import random
import numpy as np
import torch
import torch.utils.data
import board
import player
import game

# our goal is to maximise chances of neural network winning or at least forcing a draw on a opponent
# often this task is substutied by network learning to estimate quality of moves and selecting best ones on each step
# but quality of move is poorly defined property while only game history and outcome are actually reliable
# Sauble labeled each game history by its outcome but such data is highly inconsistent
# for example many games can have similar start yet different outcomes due to following moves
# cross-entropy loss does not alleviate this issue because we expect definite outcome instead of actual uncertainty
# this leads to slowly converging learning which is only effective due to mini-batches and pure luck


# in order to play the game neural network has to learn to estimate quality of each move
# the problem is


# gathering data through running games between dumb players
player1 = player.dumb_player()
player2 = player.dumb_player()
player1_won = 0
player2_won = 0
draw = 0
games_total = 10000
games = []
for _ in range(games_total):
    dumb_game = game.game(player1, player2)
    outcome = dumb_game.run()
    if outcome == 0: draw += 1
    elif outcome == 1: player1_won += 1
    else: player2_won += 1

    games.append(dumb_game)

print("Dumb vs dumb")
print("Player 1 won:", player1_won / games_total)
print("Player 2 won:", player2_won / games_total)
print("Draw:", draw / games_total)


# training probabilistic player
probabilistic_player = player.probabilistic_player()
probabilistic_player.train(games)

# probabilistic vs dumb
dumb_player_won = 0
probabilistic_player_won = 0
draw = 0
for _ in range(games_total):
    nn_game = game.game(probabilistic_player, player2)
    outcome = nn_game.run()
    if outcome == 0: draw += 1
    elif outcome == 1: probabilistic_player_won += 1
    else: dumb_player_won += 1

print("\nProbabilistic vs dumb")
print("Probabilistic player won:", probabilistic_player_won / games_total)
print("Dumb player won:", dumb_player_won / games_total)
print("Draw:", draw / games_total)

# dumb vs propabilistic
dumb_player_won = 0
probabilistic_player_won = 0
draw = 0
for _ in range(games_total):
    nn_game = game.game(player1, probabilistic_player)
    outcome = nn_game.run()
    if outcome == 0: draw += 1
    elif outcome == 1: dumb_player_won += 1
    else: probabilistic_player_won += 1

print("\nDumb vs probabilistic")
print("Dumb player won:", dumb_player_won / games_total)
print("Probabilistic player won:", probabilistic_player_won / games_total)
print("Draw:", draw / games_total)
