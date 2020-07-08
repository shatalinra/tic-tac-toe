# tic-tac-toe

Neural network for playing tic-tac-tie. Inspired by Daniel Saubel [article](https://medium.com/swlh/tic-tac-toe-and-deep-neural-networks-ea600bc53f51) and [repository](https://github.com/djsauble/tic-tac-toe-ai) but using completely different approach. 

## Results

Results for algorithms including Saubel's neural network are presented in this table.

| Algorithm    | Wins when plays first | Draws when plays first | Loss when plays first | Wins when plays second | Draws when plays second | Loss when plays second |
| ------------ | --------------------- | ---------------------- | --------------------- | ---------------------- | ----------------------- | ---------------------- |
| Saubel       | 95.0%                 | 3.4%                   | 1.6%                  | 42.7%                  | 53.0%                   | 4.3%                   |
| Probalistic  | 97.8%                 | 2.16%                  | 0.0%                  | 62.85%                 | 21.92%                  | 15.23%                 |

Propabilistic algorithm is great at winning games but have problems with forcing a draw while playing second. Saubel's network have lesser chances of winning which can be explained by using inconsistend training data leading to troubled convergence and quite high loss.
