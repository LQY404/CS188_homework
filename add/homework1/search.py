# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
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
    
import searchAgents
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # print(problem.getStartState())
    from game import Directions
    import searchAgents
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST
    stop = Directions.STOP
    
    #print(problem.getStartState())
    #print(problem.walls.data)
    #print(type(problem.walls.data))
  
    print(type(problem))
    #problem_type = type(searchAgents.PositionSearchProblem).__name__
    #print(problem_type == searchAgents.PositionSearchProblem)
    print(isinstance(problem, searchAgents.PositionSearchProblem))
    #assert 1 == 0
    start_pos = problem.getStartState()
    
    #print(start_pos[0], start_pos[1])
    #print(problem.walls[start_pos[0]][start_pos[1]])
    
    #print("目标：", problem.goal)
    
    #next_pos = (start_pos[0]+1, start_pos[1]+0)
    
    #print(problem.isGoalState(next_pos))
    
    #h = problem.walls.height # wall属性被屏蔽了
    #w = problem.walls.width
    #book = [[0 for _ in range(h)] for _ in range(w)] 
    
    actions = []
    #from util import Queue
    #actions = Queue()
    #find = DFS_v2(problem, actions, start_pos)
    #print(find)
    #return actions
    DFS_v3(problem)
    
from game import Directions

def DFS_v3(problem):
    book = util.Stack()
    start = [problem.getStartState(), 0, []]
    book.push(start)
    closed = []
    while not book.isEmpty():
        [state, cost, path] = book.pop()
        if problem.isGoalState(state):
            return path
        if not state in closed:
            closed.append(state)
            for child_state, child_action, child_cost in problem.expand(state):
                new_cost = cost + child_cost
                new_path = path + [child_action]
                book.push([child_state, new_cost, new_path])

# not general
def DFS_v2(problem, actions, state):

    flag = False
    problem.expand(state)
    
    if problem.isGoalState(state):
        #return True
        flag = True
        return flag
    
    next_actions = problem.getActions(state)
    #print(problem.expand(state))
    #print("已经走过了", problem.getExpandedStates())
    for d in next_actions:
        nstate = problem.getNextState(state, d)
        # 不需要判断是否越界，getActions函数内部处理
        
        if isinstance(problem, searchAgents.PositionSearchProblem):
            if nstate in problem._visitedlist:
                continue
        else:
            if nstate in problem.getExpandedStates():
                continue
        
        
        #print("from " + str(state) + " to " + str(nstate))
        if problem.isGoalState(nstate):  # 需要提前判断，因为根据评分的函数，我们发现最终的路径中不能包含终点（所以提前返回），但是路径需要包含（所以还需要action操作）
            actions.append(d)
            flag = True
            break
            
        actions.append(d)
        flag = flag or DFS_v2(problem, actions, nstate)
        if flag:
            break
            #return True
        
        actions.pop()
        
    return flag
    
def DFS_v1(problem, actions, book, state):
    
    book[state[0]][state[1]] = 1  # 标识
    if problem.isGoalState(state):
        #print("找到目标")
        return True
    #    print("找到目标")
    #    return actions
    
    h = problem.walls.height
    w = problem.walls.width
    
    #print("start from ", state)
    for d in [(1, 0), (0, -1), (-1, 0), (0, 1)]:
        
        #print(d)
        nstate = (state[0] + d[0], state[1] + d[1])
        #print("可能的方向:", nstate)  # 可能的方向
        
        if nstate[0]<0 or nstate[1]<0 or nstate[0]>=w or nstate[1]>=h:  # 可能会有Bug
            #print(str(nstate) + "越界")
            continue
            
        if problem.walls[nstate[0]][nstate[1]]:
            #print(str(nstate) + str("处是墙壁"))
            continue
        
        if book[nstate[0]][nstate[1]] == 1:
            #print(str(nstate) + "已经走过了")
            continue
        
        #print("from " + str(state) + " to " + str(nstate))
        
        #assert 1 == 0
        actions.append(d)
        flag = myDFS(problem, actions, book, nstate)
        if flag:
            return True
            
        book[nstate[0]][nstate[1]] = 0  # 取消标记
        actions.pop()
        
    
    #return False
import eightpuzzle

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    from util import Queue
    
    start_pos = problem.getStartState()
    #print(type(problem))
    #print("起点", start_pos)
    #print(isinstance(problem, eightpuzzle.EightPuzzleSearchProblem))
    #print("终点", problem.goals) if not isinstance(problem, searchAgents.PositionSearchProblem) else print("终点", problem.goal)
    actions = {}
    ractions = {}
    
    book = Queue()
    #actions.append(start_pos)
    #problem.expand(start_pos)
    start = [start_pos, 0, []]
    book.push(start)
    close = []
    
    
    #print("开始搜寻")
    #flag = problem.isGoalState(start_pos)
    flag = False
    rdirs = {
        'South': 'North',
        'West': 'East',
        'East': 'West',
        'North': 'South'
    }
    goal = None
    
    while not book.isEmpty():
        [state, cost, path] = book.pop()
        if problem.isGoalState(state):
            return path
        
        if state in close:
            continue
        
        close.append(state)
        for nstate, naction, ncost in problem.expand(state):
            next_cost = cost + ncost
            next_path = path + [naction]
            book.push([nstate, next_cost, next_path])
            
    
    
    
def breadthFirstSearch2(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    from util import Queue
    
    start_pos = problem.getStartState()
    #print(type(problem))
    #print("起点", start_pos)
    #print(isinstance(problem, eightpuzzle.EightPuzzleSearchProblem))
    #print("终点", problem.goals) if not isinstance(problem, searchAgents.PositionSearchProblem) else print("终点", problem.goal)
    actions = {}
    ractions = {}
    
    book = Queue()
    #actions.append(start_pos)
    #problem.expand(start_pos)
    book.push(start_pos)
    
    #print("开始搜寻")
    #flag = problem.isGoalState(start_pos)
    flag = False
    rdirs = {
        'South': 'North',
        'West': 'East',
        'East': 'West',
        'North': 'South'
    }
    goal = None
    while not book.isEmpty():
        state = book.pop()
        if isinstance(problem, searchAgents.PositionSearchProblem):
            if state in problem._visitedlist:
                continue
        else:
            if state in problem.getExpandedStates():
                continue
                    
        actions[state] = {}
        
        nactions = problem.getActions(state)
        
        ex_flag = True
        for d in nactions:
            nstate = problem.getNextState(state, d)
            
            if isinstance(problem, searchAgents.PositionSearchProblem):
                if nstate in problem._visitedlist:
                    continue
            else:
                if nstate in problem.getExpandedStates():
                    continue
            
            if nstate in book.list:
                ex_flag = False
                continue
            
            ractions[nstate] = {}
            
            ractions[nstate][d] = state # 到时候反向即可
            #print("from " + str(state) + " to " + str(nstate))
            actions[state][d] = nstate
            if problem.isGoalState(nstate):
                #actions.append(d)
                #actions[state][d] = nstate
                flag = True
                goal = nstate
                #break  #不能直接break

                
            
            #actions.append(d)
            #actions[state][d] = nstate
            book.push(nstate)
            
        if state != goal:
            problem.expand(state)
        
        
        if flag:
            #goal = state
            while not book.isEmpty():
                s = book.pop()
                if s != goal:
                    problem.expand(s)
                else:
                    break  # 到目标为止
            break
            
    assert goal is not None
    #print(flag)
    #print(book.list)
    #print(actions)
    #for k, v in actions.items():
    #    print(k)
    #    print(v)
    #print("反向")
    #print(ractions)
    #for k, v in ractions.items():
    #    print(k)
    #    print(v)
    #print(list(ractions.keys())[-1])
    nactions = []
    #s = list(ractions.keys())[-1]
    s = goal
    #assert 1 == 0
    #print(s)
    if isinstance(problem, searchAgents.PositionSearchProblem):
    
        while s[0] != start_pos[0] or s[1] != start_pos[1]:
            dd = ractions[s]
            da = list(dd.keys())[-1]
            ds = list(dd.values())[-1]
            #print(str(s) + " to " + str(ds))
            s = ds
            nactions.append(da)
    else:
        while s != start_pos:
            dd = ractions[s]
            da = list(dd.keys())[-1]
            ds = list(dd.values())[-1]
            #print(str(s) + " to " + str(ds))
            s = ds
            nactions.append(da)
    
    
    #nactions.append()
    #print(nactions[::-1])
    #assert 1 == 0
    return nactions[::-1]
    
    
    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

#from searchAgents import manhattanHeuristic

def aStarSearch(problem, heuristic=nullHeuristic):
    fringe = util.PriorityQueue()
    start = [problem.getStartState(), 0, []]
    p = 0
    fringe.push(start, p)  # queue push at index_0
    closed = []
    while not fringe.isEmpty():
        [state, cost, path] = fringe.pop()
        # print(state)
        if problem.isGoalState(state):
            # print(path)
            return path  # here is a deep first algorithm in a sense
        if state not in closed:
            closed.append(state)
            for child_state, child_action, child_cost in problem.expand(state):
                new_cost = cost + child_cost
                new_path = path + [child_action, ]
                fringe.push([child_state, new_cost, new_path], new_cost + heuristic(child_state, problem))
                
    util.raiseNotDefined()
            
            
def aStarSearch2(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    start_pos = problem.getStartState()
    print("起点:", start_pos)
    print(type(problem)) #searchAgents.PositionSearchProblem
    #assert 1 == 0
    
    openl = []
    closel = []
    
    
    openl.append(start_pos)
    book = []
    
    ps = None
    actions = []
    print("终点", problem.goals) if not isinstance(problem, searchAgents.PositionSearchProblem) else print("终点", problem.goal)
    goals = problem.goals if not isinstance(problem, searchAgents.PositionSearchProblem) else [problem.goal]  # 可能多目标
    
    flag = False
    #print(problem.walls)
    #print(problem.children if not isinstance(problem, searchAgents.PositionSearchProblem) else "")
    
    state = None
    while len(openl) != 0:
        if len(openl) == 1 or state is None:
            state = openl.pop()
        else:
            pass
    
    
# 和A*算法类似
def LikeAStarSearch(problem):
    book = util.PriorityQueue()
    start = [problem.getStartState(), 0, []]
    p = 0
    book.push(start, p)
    closed = []
    while not book.isEmpty():
        [state, cost, path] = book.pop()
        if problem.isGoalState(state):
            return path
        if state not in closed:
            closed.append(state)
            for child_state, child_action, child_cost in problem.expand(state):
                new_cost = cost + child_cost
                new_path = path + [child_action, ]
                book.push([child_state, new_cost, new_path], new_cost)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
