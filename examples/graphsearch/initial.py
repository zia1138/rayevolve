import collections
from typing import Dict, Iterable, List, Optional

# -------------------------
# Environment Interface
# -------------------------

class SearchEnv:
    """
    Interface for the search environment.
    The evaluator will pass an instance of this class to your algorithm.
    """
    @property
    def start(self): 
        """The start node."""
        pass

    @property
    def goal(self): 
        """The goal node."""
        pass

    def is_goal(self, n) -> bool: 
        """Check if node n is the goal."""
        pass

    def neighbors(self, n) -> Iterable: 
        """Yield neighbors of node n. Costs are incurred when iterating."""
        pass


# EVOLVE-BLOCK-START

def reconstruct_path(came_from: Dict, start, goal) -> List:
    cur = goal
    path = [cur]
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def graph_search(env: SearchEnv) -> Optional[List]:
    """
    Find a path from env.start to env.goal.
    
    Args:
        env: An instance of SearchEnv.
             
    Returns:
        A list of nodes representing the path from start to goal (inclusive),
        or None if no path exists.
    """
    start = env.start
    goal = env.goal

    q = collections.deque([start])
    visited = {start}
    came_from = {}

    while q:
        cur = q.popleft()

        if env.is_goal(cur):
            if start == goal:
                return [start]
            return reconstruct_path(came_from, start, goal)

        for nbr in env.neighbors(cur):
            if nbr in visited:
                continue
            visited.add(nbr)
            came_from[nbr] = cur
            q.append(nbr)

    return None

# EVOLVE-BLOCK-END
