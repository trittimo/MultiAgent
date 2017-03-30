from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  def getAction(self, gameState):
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

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
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
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

class Expectimax:
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
      return self.exp(state, agent, depth)

  def max(self, state, agent, depth):
    action = (None, -float("inf"))
    if not state.getLegalActions(agent):
      return (None, self.evalfn(state))

    for move in state.getLegalActions(agent):
      if move == Directions.STOP:
        continue

      successor = state.generateSuccessor(agent, move)
      possible = self.getBestAction(successor, agent + 1, depth)

      maxVal = max(action[1], possible[1])

      if maxVal is not action[1]:
        action = (move, maxVal)

    return action

  def exp(self, state, agent, depth):
    action = [None, 0]
    if not state.getLegalActions(agent):
      return (None, self.evalfn(state))

    legal = state.getLegalActions(agent)
    prob = 1.0/len(legal)
    for move in legal:
      if move == Directions.STOP:
        continue

      successor = state.generateSuccessor(agent, move)
      possible = self.getBestAction(successor, agent + 1, depth)

      expVal = prob * possible[1]
      action[1] = action[1] + expVal
      action[0] = move

    return tuple(action)

class ExpectimaxAgent(MultiAgentSearchAgent):
  def getAction(self, gameState):
    expectimax = Expectimax(self.depth, self.evaluationFunction)
    return expectimax.getBestAction(gameState, 0, 0)[0]

def betterEvaluationFunction(currentGameState):
  """
    DESCRIPTION:
    I just took into account where the nearest food is and where the enemies are
    Just trying to maximize the distance to enemies and minimize the distance to
    food. I also added on the current game score, because a better game score is
    better.
  """

  pacPos = currentGameState.getPacmanPosition()
  distFn = lambda pos : manhattanDistance(pacPos, pos)

  food = [-distFn(pos) for pos in currentGameState.getFood().asList()]
  enemies = [distFn(pos) for pos in currentGameState.getGhostPositions()]
  enemies = [-1.0/dist if dist != 0 else 0 for dist in enemies]
  
  if not food:
    food.append(0)

  return max(food) + min(enemies) + currentGameState.getScore()




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

