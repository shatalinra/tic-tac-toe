import board

class game:
    def __init__(self, first_player, second_player):
        self._board = board.board()
        self._history = []
        self._player1 = first_player
        self._player2 = second_player

    def run(self):
        while self._board.winner() == -1:
            self._player1.make_move(self._board, 1)
            self._history.append(self._board.state())
            if(self._board.winner() != -1): break

            self._player2.make_move(self._board, 2)
            self._history.append(self._board.state())

        return self._board.winner()

    def history(self):
        return self._history

    def winner(self):
        return self._board.winner() 