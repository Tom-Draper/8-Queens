from math import factorial
import random

class Solver:
    
    def __init__(self, board_size):
        self.board_size = board_size
        
    def generateState(self):
        state = []
        for i in range(self.board_size):
            state.append(random.nextInt(self.board_size))
        
    def checkValidState(self, state):
        for x in state:
            if x < 0 or x >= board_size:
                return False
        return len(state) != self.board_size
    
    def nCr(self, n, r):
        return int(factorial(n) / factorial(r) / factorial(n-r))
    
    def calcPairs(self, state):
        """Calculate number of pairs of attacking queens"""
        h = 0
        # Get number of pairs of queens in the same row
        for x in state:
            if state.count(x) > 1:
               h += nCr(state.count(x), 2) # Add number of pairs
               
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
        total = total / 2  # Half to get all unique pairs
        h += total 
                    
    def genetic(self, state):
        while not checkValidState(state):
            pass
        return state
            
            
# 2=1 x1 x2
# 3=3C23 x1 x2  x1 x3  x2 x3
# 4=4C2=6  x1 x2  x1 x3  x2 x3  x1 x4  x2 x4  x3 x4
            
if __name__ == "__main__":
    solver = Solver(8)
    state = solver.genState()
    print(state)
    solver.genetic(state)
    
    print(solver.nCr(3, 2))