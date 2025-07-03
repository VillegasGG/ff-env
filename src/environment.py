import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tree_generator as tg
from tree_utils import Tree

class FFProblemEnv(gym.Env):
    def __init__(self, space_limits, num_nodes=None, root_degree=None, type_root_degree=None, node_positions=None, adjacency_matrix=None):
        '''
        Initializes the environment for the FFProblem.
        Parameters:
        - space_limits: Tuple defining the limits for the action space. 
        - num_nodes: Number of nodes in the tree (if generating a random tree).
        - root_degree: Degree of the root node.
        - node_positions: Optional positions for the nodes in the tree.
        - adjacency_matrix: Optional adjacency matrix for the tree.       
        - type_root_degree: Type of root degree ('exact' or 'min').
        
        '''

        super().__init__()

        self.render_mode = None
        
        # Create tree or use provided parameters

        # Validate input combinations
        if num_nodes is not None and (node_positions is not None or adjacency_matrix is not None):
            raise ValueError("Cannot specify both num_nodes and node_positions/adjacency_matrix")
        
        if node_positions is not None and adjacency_matrix is None:
            raise ValueError("If node_positions is provided, adjacency_matrix must also be provided")
        
        if adjacency_matrix is not None and node_positions is None:
            raise ValueError("If adjacency_matrix is provided, node_positions must also be provided")
        

        # If num_nodes is provided, generate a random tree
        if num_nodes is not None:
            tree = tg.generate_random_tree(num_nodes, root_degree, type_root_degree, space_limits)

        # If node_positions is provided, use them
        elif node_positions is not None:
            tree = tg.generate_tree_from_adjacency_matrix(adjacency_matrix, node_positions)

        else:
            raise ValueError("Either num_nodes or node_positions/adjacency_matrix must be provided")

        min_pos = space_limits[0]
        max_pos = space_limits[1]
        
        self.action_space = spaces.Box(
            low=np.array([min_pos, min_pos, min_pos], dtype=np.float32),   
            high=np.array([max_pos, max_pos, max_pos], dtype=np.float32),    
            shape=(3,),                                          
            dtype=np.float32
        )

