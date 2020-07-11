import board
import game_history

class Game:
    def __init__(self, first_player, second_player):
        self._player1 = first_player
        self._player2 = second_player
        self._board = board.Board()
        self._history = game_history.GameHistory()

    def run(self):
        while self._board.winner() == -1:
            possible_moves = self._board.possible_moves()
            choosen_move = self._player1.choose_move(possible_moves, 1, self.history())
            self._board.mark(choosen_move, 1)
            self._history.record_move(choosen_move)
            if(self._board.winner() != -1): break

            possible_moves = self._board.possible_moves()
            choosen_move = self._player2.choose_move(possible_moves, 2, self.history())
            self._board.mark(choosen_move, 2)
            self._history.record_move(choosen_move)

        self._history.record_outcome(self._board.winner())
        return self._board.winner()

    def length(self):
        return len(self._history.moves())

    def winner(self):
        return self._board.winner()

    def history(self):
        return self._history