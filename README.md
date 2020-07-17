# tic-tac-toe

Algorithms for playing tic-tac-tie. Inspired by Daniel Saubel [article](https://medium.com/swlh/tic-tac-toe-and-deep-neural-networks-ea600bc53f51) and [repository](https://github.com/djsauble/tic-tac-toe-ai) but using variety of approaches.

## Algorithms

**Dumb player**: uses random for choosing one of the possible moves. Basic player for gathering some training data about game and testing other algorithms.

**Probabilistic**: during training accumulates statistics about how often some board state led to certain outcome. Chooses move by selecting move with highest score to win or draw. If no similar board state was found, resorts to random choosing of move.

**Dense autoencoder**: neural network trained to reconstruct game history described by dense feature - outcome and number of turn on which each cell was taken. Such representation itself does not allow overtaking cell or making multiple moves by same player which lifts burden of learning rules from network. Yet its learning fails to properly converge and current best variation of the model compresses 10 discrete inputs to only 7 continues values while reconstructing only 20% of training data right and playing not better than random. Tried ReLU instead of tanh, adding layers, dividing training data on mini-batches but nothing seems to work. Rest of the code seems fine because using just one hidden layer with 9 neurons leads to 100% reconstruction of training data while not learning anything useful.   

**Sparse autoencoder**: neural network trained to reconstruct game history described by sparse feature - outcome, length of game and binary arrays for empty, cross and nought cells. Such representation allows violating game rules but binary arrays seemed to be better fit for convolutional neural networks. Although several convolutional and deconvolutional layers were succesfully utilized, the latent space is still very large (82 values) and adding fully connected layers break 100% reconstruction of training data. Large latent space means that no good compression of input was reached and predictably neutwork fails to impute game future states, leading to 100% random moves and playing same as dumb player.

## Results

Results for algorithms playing against dumb player including Saubel's neural network are presented in this table.

| Algorithm     | Wins when plays first | Draws when plays first | Loss when plays first | Wins when plays second | Draws when plays second | Loss when plays second |
| ------------- | --------------------- | ---------------------- | --------------------- | ---------------------- | ----------------------- | ---------------------- |
| Saubel        | 95.0%                 | 3.4%                   | 1.6%                  | 42.7%                  | 53.0%                   | 4.3%                   |
| Probabilistic | 97.8%                 | 2.16%                  | 0.0%                  | 62.85%                 | 21.92%                  | 15.23%                 |

Probabilistic algorithm is great at winning games but have problems forcing a draw while playing second. Saubel's network have lesser chances of winning which can be explained by using inconsistend training data leading to troubled convergence and quite high loss but still forces more draws while playing second.
