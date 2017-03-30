# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # if power-pellet nearby, go there
    # look at food-grid

    from game import Actions
    from math import floor
    walls = successorGameState.getWalls()
    enemies = [ghost.getPosition() for ghost in newGhostStates]
    enemiesPosition = [enemies] + [Actions.getLegalNeighbors(pos, walls) for pos in enemies]
    enemiesPosition = [item for sublist in enemiesPosition for item in sublist]
    enemiesPosition = [(int(p[0]), int(p[1])) for p in enemiesPosition]

    # If the new position is in a square that could possibly be occupied by an enemy
    # Don't go there
    if newPos in enemiesPosition:
      return -1000

    # Automatically go to the next square if it has food in it
    if currentGameState.getFood()[newPos[0]][newPos[1]]:
      return 1000

    # Apply a penalty for staying in the same spot
    if newPos == currentGameState.getPacmanPosition():
      return -500

    foodPos = []
    for x in range(newFood.width):
     for y in range(newFood.height):
       if newFood[x][y]:
         foodPos.append((x, y))
    return sum([float(1/2) ** float(manhattanDistance(newPos, foodPos)) for foodPos in newFood])

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class Minimax:
  def __init__(self, maxDepth, evalfn):
    self.maxDepth = maxDepth
    self.evalfn = evalfn

  def getBestAction(self, state, agent, depth):
    if agent >= state.getNumAgents():
      agent = 0
      depth = depth + 1

    if depth == self.maxDepth:
      return (None, self.evalfn(state))

    if agent == 0:
      return self.max(state, agent, depth)
    else:
      return self.min(state, agent, depth)

  def common(self, state, agent, depth, fn, init):
    action = (None, init)
    if not state.getLegalActions(agent):
      return (None, self.evalfn(state))

    for move in state.getLegalActions(agent):
      if move == Directions.STOP:
        continue

      successor = state.generateSuccessor(agent, move)
      possible = self.getBestAction(successor, agent + 1, depth)

      minVal = fn(action[1], possible[1])

      if minVal is not action[1]:
        action = (move, minVal)

    return action

  def min(self, state, agent, depth):
    return self.common(state, agent, depth, min, float("inf"))
    
  def max(self, state, agent, depth):
    return self.common(state, agent, depth, max, -float("inf"))

class MinimaxAgent(MultiAgentSearchAgent):
  def getAction(self, gameState):
    minimax = Minimax(self.depth, self.evaluationFunction)
    return minimax.getBestAction(gameState, 0, 0)[0]

class AlphaBeta:
  def __init__(self, maxDepth, evalfn):
    self.maxDepth = maxDepth
    self.evalfn = evalfn

  def getBestAction(self, state, agent, depth, alpha, beta):
    if agent >= state.getNumAgents():
      agent = 0
      depth = depth + 1

    if depth == self.maxDepth:
      return (None, self.evalfn(state))

    if agent == 0:
      return self.max(state, agent, depth, alpha, beta)
    else:
      return self.min(state, agent, depth, alpha, beta)

  def common(self, state, agent, depth, fns, init, alpha, beta):
    action = (None, init)
    if not state.getLegalActions(agent):
      return (None, self.evalfn(state))

    for move in state.getLegalActions(agent):
      if move == Directions.STOP:
        continue

      successor = state.generateSuccessor(agent, move)
      possible = self.getBestAction(successor, agent + 1, depth, alpha, beta)

      comp, maxminfn = fns

      correct = maxminfn(action[1], possible[1])

      if correct is not action[1]:
        action = (move, correct)

      

      if maxminfn == max:
        if comp(action[1], beta):
          return action
        alpha = maxminfn(alpha, action[1])
      else:
        if comp(action[1], alpha):
          return action
        beta = maxminfn(beta, action[1])

    return action

  def min(self, state, agent, depth, alpha, beta):
    import operator
    return self.common(state, agent, depth, (operator.le, min), float("inf"), alpha, beta)
    
  def max(self, state, agent, depth, alpha, beta):
    import operator
    return self.common(state, agent, depth, (operator.ge, max), -float("inf"), alpha, beta)

class AlphaBetaAgent(MultiAgentSearchAgent):
  def getAction(self, gameState):
    alphabeta = AlphaBeta(self.depth, self.evaluationFunction)
    return alphabeta.getBestAction(gameState, 0, 0, -float("inf"), float("inf"))[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

