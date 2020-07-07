import random
import numpy as np
import torch
import board
import player

currentBoard = board.board()

player1 = player.dumb_player()
player2 = player.dumb_player()
while currentBoard.winner() == -1:
    player1.makeMove(currentBoard, 1)
    if(currentBoard.winner() != -1): break
    player2.makeMove(currentBoard, 2)

print(currentBoard.state())
print(currentBoard.possible_moves())
print(currentBoard.winner())
currentBoard.print()