# 8-Queens
Solves the 8 Queens puzzle using a variety of local search algorithms.

## The 8-Queens Puzzle
The 8-Queens puzzle is the problem of placing 8 chess queens on an 8x8 chessboard such that no queen can attack any other. There are 64C8 = 4,426,165,368 permutations of queen arrangements with 92 distinct solutions. Excluding solutions that differ by rotations, reflections, the puzzle has only 12 solutions.
https://en.wikipedia.org/wiki/Eight_queens_puzzle

### Algorithms implemented:
- Simulated annealing search
- Local beam search
- Stochastic beam search
- Genetic algorithm

All algorithms measure their current progress using the current amount of attacking queen pairs and try to minimise this number. Each algorithm iteratively modified its current state(s) and generally moves to states with a lower amount of attacking pairs. What often occurs is that only moving to improved states can lead to dead ends or result in the algorithm having to finish on a local maximum, not a the global maximum (a solution). As a result, the algorithms have to occassionaly move to a worse state in order to back track out of these situations. This is achieved probabilistically by accepting a worse state p percent of the time.

## Simulated Annealing
This algorithm begins on a random starting state and generates all successors to that state. It considers a random successor and accepts this state if it improves upon the state it already had. But if the selected successor state happens to be worse that the current state, we accept it with probability p. This is repeated until a solution is found.  

## Local Beam
This algorithm builds upon the simulated annealing algorithm but instead of a single state, it stores and processes k number of states in parallel. It generates successors to these k states and selects the top k states from the complete list. This speeds up the chance of finding a goal state compared to simulated annealing.

## Stochastic Beam
This algorithm is similar to local beam. Instead of choosing the top k states, it chooses successors randomly, with a bias towards the better performing ones. 

## Genetic Algorithm
This algorithm iteratively takes a population of states of size k and orders them by their fitness. It then mixes the states in pairs to create new states, with the worst performing state left out from mixing. There is then a mutation phase where state values are randomly changed with a set probability.
