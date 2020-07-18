import copy
import random
import numpy as np
import torch
import torch.utils.data
import board
import game
import players
from players import dumb, probabilistic, dense_autoencoder, sparse_autoencoder, simple, human

def print_autoencoder_metrics(title, autoencoder):
    print(title, "random moves:", 100 * autoencoder.random_moves_ratio(), "%\n")

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


# gathering data through running games between dumb players
player1 = players.dumb.Player()
player2 = players.dumb.Player()
dumb_games = run_games(10000, player1, "dumb1", player2, "dumb2")

# create additional data by adding equivalent games
original_games_count = len(dumb_games)
for i in range(original_games_count):
    dumb_games += dumb_games[i].equivalent_games()

# loading/training probabilistic player
probabilistic_player = players.probabilistic.Player()
try:
    probabilistic_player.load("models/probalistic.pickle")
except FileNotFoundError:
    probabilistic_player.train(dumb_games, "models/probalistic.pickle")

run_games(1000, probabilistic_player, "probalistic", player2, "dumb")
run_games(1000, player1, "dumb", probabilistic_player, "probalistic")

# loading/training simple player
simple_player = players.simple.Player()
try:
    simple_player.load("models/simple.pt")
except FileNotFoundError:
    simple_player.train(dumb_games, "models/simple.pt")

run_games(1000, simple_player, "simple player", player2, "dumb")
run_games(1000, player1, "dumb", simple_player, "simple player")

# loading/training dense autoencoder player
dense_autoencoder_player = players.dense_autoencoder.Player()
try:
    dense_autoencoder_player.load("models/dense_autoencoder.pt")
except FileNotFoundError:
    dense_autoencoder_player.train(dumb_games, "models/dense_autoencoder.pt")

run_games(1000, dense_autoencoder_player, "dense autoencoder", player2, "dumb")
print_autoencoder_metrics("dense autoencoder", dense_autoencoder_player)
dense_autoencoder_player.reset_metrics()

run_games(1000, player1, "dumb", dense_autoencoder_player, "dense autoencoder")
print_autoencoder_metrics("dense autoencoder", dense_autoencoder_player)
dense_autoencoder_player.reset_metrics()

# loading/training sparse autoencoder player
sparse_autoencoder_player = players.sparse_autoencoder.Player()
try:
    sparse_autoencoder_player.load("models/sparse_autoencoder.pt")
except FileNotFoundError:
    sparse_autoencoder_player.train(dumb_games, "models/sparse_autoencoder.pt")

run_games(1000, sparse_autoencoder_player, "sparse autoencoder", player2, "dumb")
print_autoencoder_metrics("sparse autoencoder", sparse_autoencoder_player)
sparse_autoencoder_player.reset_metrics()

run_games(1000, player1, "dumb", sparse_autoencoder_player, "sparse autoencoder")
print_autoencoder_metrics("sparse autoencoder", sparse_autoencoder_player)
sparse_autoencoder_player.reset_metrics()

# using best model to play against human
human_player = human.Player()
message = ["Draw!", "Network won!", "You won!"]
for _ in range(10):
    demo_game = game.Game(simple_player, human_player)
    demo_game.run()
    board.print_board_state(demo_game.history().states()[-1])
    print(message[demo_game.winner()])
