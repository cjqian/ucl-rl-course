"""maze-solver.py: Gives a path through an input maze."""
"""
   Exercise adapted from Introduction to Reinforcement Learning lectures:
   http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/intro_RL.pdf
"""
__author__ = 'cjqian@'

import sys
import unittest
from collections import deque

""" Coordinate vectors. """
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

class Direction:
  PRINT_MAP = {UP: '^', DOWN: 'v', LEFT: '<', RIGHT: '>'}

  def __init__(self, coord):
    self.coord = coord

  @staticmethod
  def add(x, direction):
    return (x[0] + direction[0], x[1] + direction[1])

""" Boards take in a MxN grid with 0s use to denote invalid spaces. """
class Board:
  def __init__(self, grid):
    self.grid = grid

    self.length = len(grid)
    if self.length == 0:
      raise ValueError('Board must have at least one square.')

    self.width = len(grid[0])
    if self.width == 0:
      raise ValueError('Grid must have at least one square.')

  def canMoveTo(self, coord):
    y = coord[0]
    x = coord[1]

    return self.grid[coord[0]][coord[1]] != 0 if self.__inBounds__(
        coord) else False

  def __inBounds__(self, coord):
    return coord[0] < self.length and coord[0] >= 0 and coord[
        1] < self.width and coord[1] >= 0

""" A maze has a board and an end coordinate on the maze.

    The valid move space of squares must be one connected component.
    The end coordinate must be a valid move square.

    To solve the maze from a given start coordinate,
    a path is printed from the start coordinate to the end coordinate.
"""
class Maze:
  def __init__(self, board, end):
    self.board = board

    # Checks validity of end location
    self.end = end
    if board.grid[end[0]][end[1]] == 0:
      raise ValueError('End location is invalid.')

    self.__genValFn__()
    self.__genPolicy__()

  # Generates a value function with a -1 reward per time step
  def __genValFn__(self):
    # Value function is represented as a dictionary,
    # with the value at key s being the value of the policy at that state/square
    self.valueFn = {self.end: -1}
    frontier = deque([self.end])

    visited = set()

    while len(frontier) > 0:
      coord = frontier.popleft()
      visited.add(coord)

      # Update values of all neighbors
      for d in DIRECTIONS:
        neighbor = Direction.add(coord, d)
        if neighbor not in visited and self.board.canMoveTo(neighbor):
          self.valueFn[neighbor] = self.valueFn[coord] - 1
          frontier.append(neighbor)

  # Generates a policy: our policy is to check neighbors and move
  # from each square to its lowest-value neighbor
  def __genPolicy__(self):
    self.policy = {}

    for coord in self.valueFn.keys():
      minv = -sys.maxint - 1
      for d in DIRECTIONS:
        neighbor = Direction.add(coord, d)
        if neighbor in self.valueFn:
          if self.valueFn[neighbor] > minv:
            minv = self.valueFn[neighbor]
            self.policy[coord] = d

  # From a given start coordinate (row, column), returns string representation
  # of a path to the end coordinate. For example, "> v ^ v > >".
  # Will return None if no valid path.
  def solvePath(self, start):
    if start not in self.policy:
      return None

    result = []
    cur = start
    while cur != self.end and cur in self.policy:
      result.append(Direction.PRINT_MAP[self.policy[cur]])
      cur = Direction.add(cur, self.policy[cur])

    return ' '.join(result)

class TestMazeSolver(unittest.TestCase):
  def setUp(self):
    # 0 1 0
    # 1 1 1
    # 0 0 1
    self.boardA = Board([[0, 1, 0], [1, 1, 1], [0, 0, 1]])
    self.mazeA = Maze(self.boardA, (2, 2))

    # https://screenshot.googleplex.com/LJFPXxf0ZpT
    self.boardB = Board([[0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 0],
                         [1, 1, 0, 0, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1, 1, 0],
                         [0, 0, 1, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0, 1, 1, 0],
                         [0, 1, 1, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]])
    self.mazeB = Maze(self.boardB, (6, 7))

  def testValueFunctions(self):
    #  X -4  X
    # -4 -3 -2
    #  X  X -1
    self.assertEqual(self.mazeA.valueFn, {
        (1, 2): -2,
        (0, 1): -4,
        (1, 0): -4,
        (1, 1): -3,
        (2, 2): -1
    })

    # https://screenshot.googleplex.com/5H8NMmSUJaL
    self.assertEqual(
        self.mazeB.valueFn, {
            (1, 1): -14,
            (1, 2): -13,
            (1, 3): -12,
            (1, 4): -11,
            (1, 5): -10,
            (1, 6): -9,
            (2, 0): -16,
            (2, 1): -15,
            (2, 4): -12,
            (2, 6): -8,
            (3, 1): -16,
            (3, 2): -17,
            (3, 5): -6,
            (3, 6): -7,
            (4, 2): -18,
            (4, 3): -19,
            (4, 5): -5,
            (5, 1): -24,
            (5, 3): -20,
            (5, 5): -4,
            (5, 6): -3,
            (6, 1): -23,
            (6, 2): -22,
            (6, 3): -21,
            (6, 4): -22,
            (6, 6): -2,
            (6, 7): -1,
        })

  def testPolicies(self):
    # X v X
    # > > v
    # X X ^
    # Note: policy at the end point is unimportant,
    # so we just check subset of relevant squares.
    self.assertTrue(
        set({
            (1, 2): DOWN,
            (0, 1): DOWN,
            (1, 0): RIGHT,
            (1, 1): RIGHT,
        }).issubset(set(self.mazeA.policy)))

    # https://screenshot.googleplex.com/0EXSsKpi3H1
    self.assertTrue(
        set({
            (1, 1): RIGHT,
            (1, 2): RIGHT,
            (1, 3): RIGHT,
            (1, 4): RIGHT,
            (1, 5): RIGHT,
            (1, 6): DOWN,
            (2, 0): RIGHT,
            (2, 1): UP,
            (2, 4): UP,
            (2, 6): DOWN,
            (3, 1): UP,
            (3, 2): LEFT,
            (3, 5): DOWN,
            (3, 6): LEFT,
            (4, 2): UP,
            (4, 3): LEFT,
            (4, 5): DOWN,
            (5, 1): DOWN,
            (5, 3): UP,
            (5, 5): RIGHT,
            (5, 6): DOWN,
            (6, 1): RIGHT,
            (6, 2): RIGHT,
            (6, 3): UP,
            (6, 4): LEFT,
            (6, 6): RIGHT,
            (6, 7): RIGHT
        }).issubset(set(self.mazeB.policy)))

  def testPaths_MazeA(self):
    # X v X
    # > > v
    # X X ^
    self.assertIsNone(self.mazeA.solvePath((0, 0)))
    self.assertEqual(self.mazeA.solvePath((1, 1)), '> v')
    self.assertEqual(self.mazeA.solvePath((1, 0)), '> > v')
    self.assertEqual(self.mazeA.solvePath((0, 1)), 'v > v')

  def testPaths_MazeB(self):
    # https://screenshot.googleplex.com/JC57j6jPi1j
    self.assertIsNone(self.mazeB.solvePath((1, 0)))
    self.assertIsNone(self.mazeB.solvePath((2, 3)))
    self.assertEqual(
        self.mazeB.solvePath((2, 0)), '> ^ > > > > > v v < v v > v >')
    self.assertEqual(self.mazeB.solvePath((2, 4)), '^ > > v v < v v > v >')
    self.assertEqual(
        self.mazeB.solvePath((6, 1)),
        '> > ^ ^ < ^ < ^ ^ > > > > > v v < v v > v >')
    self.assertEqual(
        self.mazeB.solvePath((6, 4)),
        '< ^ ^ < ^ < ^ ^ > > > > > v v < v v > v >')

if __name__ == '__main__':
  unittest.main()
