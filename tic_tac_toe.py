import copy
import random
import numpy as np
import torch
import torch.utils.data
import board
import game
import players
from players import dumb, probabilistic, dense_autoencoder 

def print_autoencoder_metrics(title, autoencoder):
    print(title)
    print("\tRandom moves:", 100 * autoencoder.random_moves_ratio(), "%\n")

def run_games(count, player1, player1_name, player2, player2_name):
    player1_won = 0
    player2_won = 0
    draw = 0
    stories = []
    for _ in range(count):
        new_game = game.Game(player1, player2)
        outcome = new_game.run()
        if outcome == 0: draw += 1
        elif outcome == 1: player1_won += 1
        else: player2_won += 1

        stories.append(new_game.history())

    print(player1_name, "vs", player2_name)
    print(player1_name, "won:", 100 * player1_won / count, "%")
    print(player2_name, "won:", 100 * player2_won / count, "%")
    print("Draw:", 100 * draw / count, "%\n")
    return stories

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
player1 = players.dumb.Player()
player2 = players.dumb.Player()
dumb_games = run_games(10000, player1, "dumb1", player2, "dumb2")

# loading/training probabilistic player
probabilistic_player = players.probabilistic.Player()
try:
    probabilistic_player.load("models/probalistic.pickle")
except FileNotFoundError:
    probabilistic_player.train(dumb_games, "models/probalistic.pickle")

# testing
run_games(1000, probabilistic_player, "probalistic", player2, "dumb")
run_games(1000, player1, "dumb", probabilistic_player, "probalistic")

# loading/training dense autoencoder player
dense_autoencoder_player = players.dense_autoencoder.Player()
try:
    dense_autoencoder_player.load("models/dense_autoencoder.pt")
except FileNotFoundError:
    dense_autoencoder_player.train(dumb_games, "models/dense_autoencoder.pt")

# testing
run_games(1000, dense_autoencoder_player, "dense autoencoder", player2, "dumb")
print_autoencoder_metrics("dense autoencoder", dense_autoencoder_player)
dense_autoencoder_player.reset_metrics()

run_games(1000, player1, "dumb", dense_autoencoder_player, "dense autoencoder")
print_autoencoder_metrics("dense autoencoder", dense_autoencoder_player)
dense_autoencoder_player.reset_metrics()