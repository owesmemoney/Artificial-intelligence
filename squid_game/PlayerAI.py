import numpy as np
import random
import time
import sys
import os 
from BaseAI import BaseAI
from Grid import Grid


class PlayerAI(BaseAI):

    def __init__(self) -> None:
        # You may choose to add attributes to your player - up to you!
        super().__init__()
        self.pos = None
        self.player_num = None
    
    def getPosition(self):
        return self.pos

    def setPosition(self, new_position):
        self.pos = new_position 

    def getPlayerNum(self):
        return self.player_num

    def setPlayerNum(self, num):
        self.player_num = num

    def getMove(self, grid: Grid) -> tuple:
        _, bestMove = self.minimax_move(5, float("-inf"), float("inf"),  True, grid) 
        return bestMove
    
    def getTrap(self, grid : Grid) -> tuple: 
        _, bestTrap = self.minimax_trap(5, float("-inf"), float("inf"), True, grid) 
        return bestTrap


    def utility(self, grid): # think about whose turn
        opponent_neighbors = grid.get_neighbors(grid.find(3-self.player_num), only_available=True)
        player_neighbors = grid.get_neighbors(grid.find(self.player_num), only_available=True)
        
     #   if len(player_neighbors) > 4:
      #      return 2.5*len(player_neighbors)
        

        return 2*len(player_neighbors)
    
    def utility_trap(self, grid):
        opponent_neighbors = grid.get_neighbors(grid.find(3-self.player_num), only_available=True)
        player_neighbors = grid.get_neighbors(grid.find(self.player_num), only_available=True)

        if player_neighbors <= 1:
            return -100
        
        else:
            return 3*len(player_neighbors) - len(opponent_neighbors)


    
    def terminal(self, grid): # think about whose turn, and change accordingly
        opponent_neighbors = grid.get_neighbors(grid.find(3-self.player_num), only_available=True)
        player_neighbors = grid.get_neighbors(grid.find(self.player_num), only_available=True)
        if len(opponent_neighbors) == 0 or len(player_neighbors) == 0:
            return True

    def minimax_move(self, depth, alpha, beta, maximizingPlayer, grid):
        if depth == 0 or self.terminal(grid):
            return self.utility(grid), grid.find(self.player_num)
        
        if maximizingPlayer:
            value = float("-inf")
            best_move = None
            available_moves = grid.get_neighbors(grid.find(self.player_num), only_available=True)

            for child_pos in available_moves:
                grid_copy = grid.clone()
                grid_copy.move(child_pos, self.player_num)
                new_value, _ = self.minimax_move(depth-1, alpha, beta, False, grid_copy)
                if value < new_value:
                    value = new_value
                    best_move = child_pos
                
                if value >= beta:
                    break
                alpha = max(value, alpha)                  
            return value, best_move
        
        else:
            value = float("inf")
            available_moves = grid.get_neighbors(grid.find(self.player_num), only_available = True) 
            for child_pos in available_moves:
                grid_copy = grid.clone()
                trap = self.throw(grid.find(3-self.player_num), grid_copy, child_pos) # AI is throwing to players
                grid_copy.trap(trap)
                new_value, _  = self.minimax_move(depth-1, alpha, beta, True, grid_copy)
                if value > new_value:
                    value = new_value
                    best_move = child_pos
                
                if value <= alpha:
                    break
                beta = min(beta, value)
            return value, best_move




    def minimax_trap(self, depth,  alpha, beta, maximizingPlayer, grid):
        if depth == 0 or self.terminal(grid):
            return self.utility(grid), grid.find(self.player_num)
        
        if maximizingPlayer:
            value = float("-inf")
            best_move = None
            available_moves = grid.get_neighbors(grid.find(3-self.player_num), only_available=True)

            for child_pos in available_moves:
                grid_copy = grid.clone()
                trap = self.throw(grid.find(self.player_num), grid_copy, child_pos)
                grid_copy.trap(trap)
                new_value, _ = self.minimax_trap(depth-1, alpha, beta, False, grid_copy)
                if value < new_value:
                    value = new_value
                    best_move = child_pos
                
                if value >= beta:
                    break
                alpha = max(value, alpha)   
                        
            return value, best_move

        else:
            value = float("inf")
            available_moves = grid.get_neighbors(grid.find(self.player_num), only_available = True) 
            for child_pos in available_moves:
                grid_copy = grid.clone()
                grid_copy.move(child_pos, 3-self.player_num)
                new_value, _  = self.minimax_trap(depth-1, alpha, beta, True, grid_copy)
                if value > new_value:
                    value = new_value
                    best_move = child_pos
                
                if value <= alpha:
                    break
                beta = min(beta, value)

            return value, best_move

    def throw(self, player_pos, grid : Grid, intended_position : tuple) -> tuple:
        '''
        Description
        ----------
        Function returns the coordinates in which the trap lands, given an intended location.

        Parameters
        ----------

        player : the player throwing the trap

        grid : current game Grid

        intended position : the (x,y) coordinates to which the player intends to throw the trap to.

        Returns
        -------
        Position (x_0,y_0) in which the trap landed : tuple

        '''
 
        # find neighboring cells
        neighbors = grid.get_neighbors(intended_position)

        neighbors = [neighbor for neighbor in neighbors if grid.getCellValue(neighbor) <= 0]
        n = len(neighbors)
        
        probs = np.ones(1 + n)
        
        # compute probability of success, p
        p = 1 - 0.05*(manhattan_distance(player_pos, intended_position) - 1)

        probs[0] = p

        probs[1:] = np.ones(len(neighbors)) * ((1-p)/n)

        # add desired coordinates to neighbors
        neighbors.insert(0, intended_position)
        
        # return 
        result = np.random.choice(np.arange(n + 1), p = probs)
        
        return neighbors[result]

def manhattan_distance(position, target):
        return np.abs(target[0] - position[0]) + np.abs(target[1] - position[1])

        

    