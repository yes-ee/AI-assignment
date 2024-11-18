# search.py
# ---------
'''
Copyright (C) Computer Science & Engineering, Soongsil University. This material is for educational uses only. 
Some contents are based on the material provided by other paper/book authors and may be copyrighted by them. 
Written by Haneul Pyeon, October 2024.
'''



"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()
    
    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    
    stack = util.Stack()

    start_state = problem.getStartState()
    route = []
    stack.push((start_state, route))
    position_bool = set()

    while not stack.isEmpty():
        current_state, route = stack.pop()

        # if current_state in position_bool:
        #     continue
    
        if problem.isGoalState(current_state):
            return route
        
        if current_state not in position_bool:
            position_bool.add(current_state)

            for successor, action, _ in problem.getSuccessors(current_state):
                if successor not in position_bool:
                    stack.push((successor, route+[action]))
    return []

    # stack = util.Stack()
    # stack.push((problem.getStartState(), []))

    # visited = set()

    # while not stack.isEmpty():
    #     state, actions = stack.pop()
    #     if problem.isGoalState(state):
    #         return actions
    #     if state not in visited:
    #         visited.add(state)
    #         for next_state, action, cost in problem.getSuccessors(state):
    #             if next_state not in visited:
    #                 stack.push((next_state, actions + [action]))

    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()

    start_state = problem.getStartState()
    route = []
    queue.push((start_state, route))
    position_bool = set()

    while not queue.isEmpty():
        current_state, route = queue.pop()

        # if current_state in position_bool:
        #     continue

        if problem.isGoalState(current_state):
            return route
        
        if current_state not in position_bool:
            position_bool.add(current_state)
            
            for successor, action, _ in problem.getSuccessors(current_state):
                if successor not in position_bool:
                    queue.push((successor, route+[action]))

    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    pqueue = util.PriorityQueue()
    start_state = problem.getStartState()
    route = []
    # (state, route, cost), cost
    pqueue.push((start_state, route, 0), 0)
    visited = set()

    while not pqueue.isEmpty():
        current_state, route_, cost = pqueue.pop()

        if current_state in visited:
            continue

        if problem.isGoalState(current_state):
            return route_

        visited.add(current_state)
        for successor, action, next_cost in problem.getSuccessors(current_state):
            new_route = route_ + [action]
            new_cost = cost + next_cost

            # if successor not in visited:
            pqueue.update((successor, new_route, new_cost), new_cost)

    return []
        
        

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    if problem is not None and hasattr(problem, 'goal') and problem.goal is not None:
        return util.manhattanDistance(state, problem.goal)
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pqueue = util.PriorityQueue()
    start_state = problem.getStartState()
    route = []
    # (state, route, path_cost, path_heuristic), cost+heuristic
    pqueue.push((start_state, route, 0, heuristic(start_state, problem)), 0+heuristic(start_state, problem))
    visited = set()

    while not pqueue.isEmpty():
        current_state, route_, cost, _ = pqueue.pop()

        if current_state in visited:
            continue

        if problem.isGoalState(current_state):
            return route_

        visited.add(current_state)
        for successor, action, next_cost in problem.getSuccessors(current_state):
            new_route = route_ + [action]
            new_cost = cost + next_cost
            new_heuristic = heuristic(successor, problem)

            # if successor not in visited:
            pqueue.update((successor, new_route, new_cost, new_heuristic), new_cost+new_heuristic)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
