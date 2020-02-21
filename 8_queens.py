from math import factorial
import random

class Solver:
    
    def __init__(self, board_size, state_split=0.5, mutation_chance=0.1):
        self.board_size = board_size
        # The proportion of the fitter state used when merging two states
        self.state_split = state_split
        
    def generateState(self):
        state = []
        for i in range(self.board_size):
            state.append(random.randint(0, self.board_size))
        return state
        
    def checkValidState(self, state):
        for x in state:
            if x < 0 or x >= self.board_size:
                return False
        return len(state) != self.board_size
    
    def checkValid(self, states):
        for idx in range(len(states)):
            if self.checkValidState(states[idx]):
                return tuple((True, idx))
        return tuple((False, -1))
    
    def nCr(self, n, r):
        return int(factorial(n) / factorial(r) / factorial(n-r))
    
    def calcPairs(self, state):
        """Calculate number of pairs of attacking queens"""
        h = 0
        # Get number of pairs of queens in the same row
        for x in state:
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
                    if i - j == x - y:
                        total += 1
        h += total / 2  # Half total to remove pairs from symmetry
        return h
        
    def fitness(self, states):
        """Orders states list by number of pairs of attacking queens"""
        rank = []
        for state in states:
            rank.append(tuple((state, self.calcPairs(state))))
        rank.sort(key=lambda x: x[1])
        
        sorted_states = [x[0] for x in rank]  # Get states in sorted order
        return sorted_states
    
    def merge(self, state1, state2):
        """Merges first half of state 1 and second half of state 2 to create a 
        new state. The size of a half is determined by state_split."""
        first_half_size = round(len(state1) * self.state_split)
        new_state = state1[:first_half_size] + state2[first_half_size:]
        return new_state
    
    def crossover(self, states):
        """Loops through pairs of states and merge them to create two new states."""
        # state1 = state1FirstHalf + state2SecondHalf
        # state2 = state2FirstHalf + state1SecondHalf
        # Size of first "half" determined by the state_split class variable
        for i in range(0, len(states), 2):
            state1 = states[i][:]
            state2 = states[i+1][:]
            state[i] = self.merge(state1, state2)
            state[i+1] = self.merge(state2, state1)
            
    def mutate(self, state):
        """Attempts to mutate each value in the state at the mutation_chance 
        class variable"""
                    
    def genetic(self):
        states = []
        
        for i in range(4):
            states.append(self.generateState())
                
        while not self.checkValid(states)[0]:
            states = self.fitness(states)
            states = self.crossover(states)
            # Mutation
            
        goal_state = states[self.checkValid(states)[1]]
            
        return goal_state


if __name__ == "__main__":
    solver = Solver(8, 0.5)
    solver.genetic()