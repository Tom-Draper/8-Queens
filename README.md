# 8-Queens
Solves the 8 Queens puzzle using a variety of local search algorithms.

Note: due to the nature of local beam search, it can often lead to a dead end that is unlikely to recover from as explained below, especially when the k value is low. If the local beam takes a while, exit with ctrl+c and restart.

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

Default parameters:
- board size: 8 (x8)
- queens: 8 (cannot be larger than board size)
- p probability of accepting a worse state: 0.3

## Local Beam
This algorithm builds upon the simulated annealing algorithm but instead of a single state, it stores and processes k number of states in parallel. It generates successors to these k states and selects the top k states from the complete list with the idea that it will speeds up the chance of finding a goal state compared to simulated annealing. As this algorithm takes the top k values, it often results in all the current states reaching a dead end (from testing often all states have a single pair of attacking queens left and no state can reach a goal state by moving a single queen). In this case, it usually takes the algorithm a very long time to backtrack, as after a while, the number of pairs of attacking queens of the top k of the successors are nearly always 1, and so it is difficult to escape. From testing, local beam either reaches the goal state almost immediately or will never locate it.   
The most effective way to increase the reliability of local beam search is to use a higher k value than 4, to hold and work with a larger number of states at once. You can also increase the p probability value to increase the chance of backtracking to worse states.

Defaults parameters:
- k number of states: 8
- board size: 8 (x8)
- queens: 8 (cannot be larger than board size)
- p probability of accepting a worse state: 0.3

## Stochastic Beam
This algorithm improves upon to local beam. Instead of choosing the top k states which can often lead to all states reaching a dead end (as explained above), it chooses successors randomly, but with a bias towards the better performing ones. This way it mixes a larger range of state fitnesses, and doesn't just select the very top. 

Defaults parameters:
- k number of states: 8
- board size: 8 (x8)
- queens: 8 (cannot be larger than board size)
- p probability of accepting a worse state: 0.3

## Genetic Algorithm
This algorithm is different from the others and does not generate successor states from a single state but instead mixes together pairs of states to create new ones. This algorithm iteratively takes a population of states of size k and orders them by their fitness. It then mixes the states in pairs to create new states, with the worst performing state left out from mixing. There is then a mutation phase where state values are randomly changed with a set probability.

Defaults parameters:
- population size: 8
- board size: 8 (x8)
- queens: 8 (cannot be larger than board size)
- state split: 0.5
- mutation chance: 0.1
