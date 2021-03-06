from math import factorial
from math import ceil
import random
import numpy as np
import time

class Solver:
    def __init__(self, board_size, queens):
        print(f"Board size: {board_size}")
        if queens > board_size:
            print(f"Queens set to: {queens}")
            queens = board_size
        else:
            print(f"Queens: {queens}")
        self.board_size = board_size
        self.queens = queens
        
    def generateState(self):
        # Generates a new random board state with the queens placed.
        
        state = []
        for i in range(self.board_size):
            state.append(random.randint(0, self.board_size - 1))
        return state
        
    def checkValidState(self, state):
        # Checks whether a state is valid with the board constraints.
        
        for x in state:
            if x < 0 or x >= self.board_size:
                return False
        return len(state) == self.board_size
    
    def checkValidStates(self, states):
        # Checks that all state in the list of states are valid and compatible 
        # with the board.
        
        for idx in range(len(states)):
            if not self.checkValidState(states[idx]):
                return False
        return True
    
    def nCr(self, n, r):
        # Calculates and returns the result of n choose r (nCr)
        return int(factorial(n) / factorial(r) / factorial(n-r))
    
    def calcPairs(self, state):
        # Calculates the number of pairs of attacking queens in the input state.
        
        h = 0
        # Get number of pairs of queens in the same row
        for x in set(state):  # Each unique value
            if state.count(x) > 1:
               h += self.nCr(state.count(x), 2) # Add number of pairs
               
        # Get number of pairs of queens on diagonals
        total = 0
        for i in range(len(state)):
            x = state[i]
            for j in range(len(state)):
                if j != i:
                    y = state[j]
                    # If difference in index of two values is equal to the difference 
                    # of their values, they are on the same diagonal line
                    if abs(i - j) == abs(x - y):
                        total += 1
        h += total / 2  # Half total to remove pairs from symmetry
        return h
        
    def checkFound(self, states):
        # Takes a list of states. Checks if each any of the states are a goal state.

        # Returns dictionary holding whether found, and if so, its index
        for idx in range(len(states)):
            if self.calcPairs(states[idx]) == 0:
                return dict({'Found': True, 'Idx': idx})
        return dict({'Found': False, 'Idx': None})
    
    def rank(self, states):
        # Orders states list by number of pairs of attacking queens.
        
        ranking = []
        for state in states:
            ranking.append(tuple((state, self.calcPairs(state))))
        ranking.sort(key=lambda x: x[1])

        return ranking


class SuccessorAlgorithm(Solver):
    """The parent class of algorithms that create successors for a single state 
    to then move to.
    SimulatedAnnealing, LocalBeam and StochasticBeam inherit from this.
    """
    
    def __init__(self, k, board_size, queens, p=0.1):
        Solver.__init__(self, board_size, queens)
        
        self.k = k  # Number of states 
        self.p = p  # Probability of accepting a worse state
        
        print(f"{k} states")
        print(f"p = {p}")
        
    def generateSuccessors(self, state):
        # Generate all possible successors to the current state by moving a single
        # queen.
        
        successors = []
        # Generate all successors to the input state by moving a single piece
        for idx in state:
            for value in range(self.board_size):
                if value != state[idx]: # Value not in this current state
                    # Replace this index with a new value
                    successor = state[:]
                    successor[idx] = value
                    successors.append(successor)   
        return successors 
    
    def start(self, selectSuccessors):
        """Starts the algorithm of a successor based algorithm.
        
        Arguments:
            selectSuccessor {function} -- a function that selects a list of 
            integers from a list. The function selects a successor state from a 
            list of possible successors.
        
        Returns:
            List of integers -- an accepting goal state.
        """
        
        found = False
        states = []
        # Stores h "rating" for a state (number of pairs of attacking queens)
        h = []
        
        for _ in range(self.k):
            state = self.generateState()
            states.append(state)
            # Get h "rating" of this state (number of pairs of attacking queens)
            h.append(self.calcPairs(state))
        
        # Loop while not found solution
        while (found := self.checkFound(states))['Found'] == False:
            successors = []
            # Get list of all successors to the current state(s)
            for i in range(len(states)):
                successors += self.generateSuccessors(states[i])
            
            # Select state(s) from entire collection of successor states
            selected = selectSuccessors(successors)
        
            for i in range(self.k):
                # If selected successor improves current solution, accept move
                if (new_h := self.calcPairs(selected[i])) < h[i]:
                    states[i] = selected[i]
                    h[i] = new_h
                elif np.random.uniform() < self.p:
                    # Still accept bad move p percent of the time
                    states[i] = selected[i]
                    h[i] = new_h
        
        goal_state = states[found['Idx']]
        return goal_state


class SimulatedAnnealing(SuccessorAlgorithm):
    """This algorithm iteratively takes a possible state and selects a random 
    successor to that state. If this successor improves the current state, it is
    accepted as a move. If this successor worsens the current state, it is 
    accepted with probability p.
    """
    
    def __init__(self, board_size=8, queens=8, p=0.3):
        # As simulated annealing only keeps track of a single state to find a 
        # successor to, k is set to 1.
        SuccessorAlgorithm.__init__(self, 1, board_size, queens, p)
        
    def randomSelect(self, successors):
        # Randomly selects a successor from a list of successors with uniform 
        # probability.
        
        states = []
        for _ in range(self.k):
            idx = (np.random.choice(range(len(successors)), replace=False))
            states.append(successors[idx])
        return states
    
    def start(self):
        """Finds a goal state (where no queen can attack another queen) using a
        simulated annealing algorithm. 
        
        Returns:
            List of integers -- an accepting goal state.
        """
        
        # Start the algorithm, passing in the random select function to select a
        # successor state randomly
        return super().start(self.randomSelect)


class LocalBeam(SuccessorAlgorithm):
    """This algorithm is similar to simulated annealing but uses k current 
    states rather than just 1.
    The algorithm iteratively takes k possible state and selects a random 
    successor to each state. If this successor improves the current state, 
    it is accepted as a move. If this successor worsens the current state, 
    it is accepted with probability p.
    """
    
    def __init__(self, k=8, board_size=8, queens=8, p=0.3):
        SuccessorAlgorithm.__init__(self, k, board_size, queens, p)
        
    def selectTop(self, successors):
        ranking = self.rank(successors)
        sorted_successors = [x[0] for x in ranking]
        # Return top k performing successor states
        return sorted_successors[:self.k]
        
    def start(self):
        """Finds a goal state (where no queen can attack another queen) using a
        simulated annealing algorithm.
        
        Returns:
            List of integers -- an accepting goal state.
        """
        
        # Start the algorithm, passing in the selectTop function to select the
        # top successors from
        return super().start(self.selectTop)


class StochasticBeam(SuccessorAlgorithm):
    """This algorithm is similar to the local beam algorithm but has a bias
    towards selecting a fitter successor rather than selecting one randomly. 
    This algorithm iteratively take k possible state and selects a
    successor to each state with a bias towards the better states. If this
    successor improves the current state, it is accepted as a move. If this 
    successor worsens the current state, it is accepted with probability p.
    """
    
    def __init__(self, k=8, board_size=8, queens=8, p=0.3):
        SuccessorAlgorithm.__init__(self, k, board_size, queens, p)
        
    def selectSuccessor(self, successors):
        # Selects a successor from the list of successors with a bias towards
        # fitter successors
        
        # Rank successors by their fitness (lower rank = better fitness)
        # Get tuple (state, rank) sorted ascending by their rank
        ranking = self.rank(successors)
        sorted_successors = [x[0] for x in ranking]  # Get states in sorted order
        # Take one over ranking as weight bias
        bias_weights = [1/(x[1] + 1) for x in ranking]
        prob = np.array(bias_weights) / np.sum(bias_weights)
        
        states = []
        for _ in range(self.k):
            choice = np.random.choice(len(prob), p=prob)
            states.append(sorted_successors[choice])
        return states
    
    def start(self):
        """Starts the algorithm. Finds a goal state (where no queen can attack 
        another queen) using a simulated annealing algorithm.
    
        Returns:
            List of integers -- an accepting goal state.
        """
        
        # Start the algorithm, passing in a successor seletion function biased 
        # towards the fitter options
        return super().start(self.selectSuccessor)


class Genetic(Solver):
    """This algorithm takes an initial population of n number of states, 
    measures their fitness, orders them. The algorithm then crosses over 
    pairs of states excluding the worst performing state in the population
    to create new states. The values in these states then are mutated 
    at a set probability.
    """
    
    def __init__(self, population=4, board_size=8, queens=8, state_split=0.5, mutation_chance=0.1):
        Solver.__init__(self, board_size, queens)
        
        self.population = population
        if self.population % 2 == 1:  # If odd, reset to default
            self.population = 4
        # The proportion of the state that is fitter used when merging two states
        self.state_split = state_split
        self.mutation_chance = mutation_chance
        
        print(f"Population size: {population}")
        print(f"Crossing split: {int(self.state_split*100)}/{int((1-self.state_split) * 100)}")
        print(f"Chance of mutation: {self.mutation_chance}")
    
    def merge(self, state1, state2):
        # Merges first half of state 1 and second half of state 2 to create a 
        # new state. The size of a "half" is determined by state_split.
        
        first_half_size = round(len(state1) * self.state_split)
        new_state = state1[:first_half_size] + state2[first_half_size:]
        return new_state
    
    def crossover(self, states):
        # Loops through pairs of states and merge them to create two new states.
        # Excludes the last state in states from merging.
        # state1 = state1FirstHalf + state2SecondHalf
        # state2 = state2FirstHalf + state1SecondHalf
        # Size of first "half" determined by the state_split class variable.add()
        
        for i in range(0, self.population, 2):
            if i+2 != len(states):  # If not at last two states
                # Merge this state with next state
                state1 = states[i][:]
                state2 = states[i+1][:]
                states[i] = self.merge(state1, state2)
                states[i+1] = self.merge(state2, state1)
            else:
                # When reach at last two states
                # Take previous state instead of next one
                # This removes the worst performing state from the poplation
                state1 = states[i-1][:]
                state2 = states[i][:]
                states[i] = self.merge(state1, state2)
                states[i+1] = self.merge(state2, state1)
        return states
    
    def fitnessSort(self, states):
        # Sorts a list of states each by their own fitness.

        # Get ranking tuples (state, rank) sorted ascending by rank
        ranking = self.rank(states)
        return [x[0] for x in ranking]  # Return states in sorted order
            
    def mutate(self, state):
        # Attempts to mutate each value in the state at the mutation_chance 
        # class variable.
        
        for idx in range(len(state)):
            rand = random.random()
            if rand < self.mutation_chance:
                # Generate new random value at this index
                state[idx] = random.randint(0, self.board_size - 1)
            
    def start(self):
        """
        Starts the algorithm. Finds a goal state (where no queen can attack 
        another queen) using a genetic algorithm.
        
        Returns:
            List of integers -- an accepting goal state.
        """
        
        found = False
        states = []
        for i in range(self.population):
            states.append(self.generateState())
        
        # Loop while not found solution
        while (found := self.checkFound(states))['Found'] == False:
            if self.checkValidStates(states):
                states = self.fitnessSort(states)  # Sort states by fitness
                states = self.crossover(states)  # Swap state halves
                for state in states:
                    self.mutate(states)  # Mutate handful of state values
            
        goal_state = states[found['Idx']]
        return goal_state


class Display:
    """Class to hold functions to display representations of states."""
    
    def printBoard(self, state):
        """Prints a representation of the state as a board to command line.
        
        Arguments:
            state {list of integers} -- the state to be printed.
        """
        
        width = 49
        print("-" * width)
        for row in range(len(state)):
            print("|", end='')
            for idx in range(len(state)):
                if row == state[idx]:
                    print("Q".center(5), end='|')
                else:
                    print(" " * 5, end='|')
            print("\n" + "-" * width + "\n", end='')


def runAlgorithm(algorithm):
    start = time.time()
    goal_state = algorithm.start()
    end = time.time()
    
    display = Display()
    print(goal_state)
    display.printBoard(goal_state)
    print("Time taken: %.4f seconds\n" % (end - start))

if __name__ == "__main__":
    print("Simmulated Annealing")
    sa = SimulatedAnnealing()
    runAlgorithm(sa)
    print("Local Beam Algorithm")
    lb = LocalBeam()
    runAlgorithm(lb)
    print("Stochastic Beam Algorithm")
    sb = StochasticBeam()
    runAlgorithm(sb)
    print("Genetic Algorithm")
    g = Genetic()
    runAlgorithm(g)
    