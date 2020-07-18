import board

class Player:
    def choose_move(self, possible_moves, own_mark, history):
        board.print_board_state(history.states()[-1])
        print("Possible moves are:")
        for i, move in enumerate(possible_moves):
            print(i,'-', move)
        while True:
            try:
                index = int(input("Choose one by entering zero-based index: "))
                if index < len(possible_moves):
                    break
                else:
                    print("Index should be less than total number of possible moves")
            except ValueError:
                print("Invalid input")
        return possible_moves[index]

