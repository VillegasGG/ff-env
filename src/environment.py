import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tree_generator as tg
from tree_utils import Tree
from fire_state import FireState
from firefighter import Firefighter

class FFProblemEnv(gym.Env):
    def __init__(self, space_limits, speed, position = None, num_nodes=None, root_degree=None, type_root_degree=None, node_positions=None, adjacency_matrix=None, root=None):
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

        self.validate_init_params(num_nodes, node_positions, adjacency_matrix)

        # If num_nodes is provided, generate a random tree
        if num_nodes is not None:
            self.tree, self.root = tg.generate_random_tree(num_nodes, root_degree, type_root_degree, space_limits)

        # If node_positions is provided, use them
        elif node_positions is not None:
            self.tree = tg.generate_tree_from_adjacency_matrix(adjacency_matrix, node_positions)
            self.root = root if root is not None else 0  # Default to 0 if root is not specified

        else:
            raise ValueError("Either num_nodes or node_positions/adjacency_matrix must be provided")

        # Initialize fire state
        self.fire_state = FireState(self.tree)

        min_pos = space_limits[0]
        max_pos = space_limits[1]


        # Initialize ff
        if position is None:
            position = np.random.uniform(low=min_pos, high=max_pos, size=(3,))
        self.ff = Firefighter(speed=speed, position=position, remaining_time=1.0)

        # Initialize ff random position
        self.ff_position = np.random.uniform(low=min_pos, high=max_pos, size=(3,))
        
        self.action_space = spaces.Box(
            low=np.array([min_pos, min_pos, min_pos], dtype=np.float32),   
            high=np.array([max_pos, max_pos, max_pos], dtype=np.float32),    
            shape=(3,),                                          
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "node_positions": spaces.Box(
                low=np.full((self.tree.nodes.shape[0], 3), min_pos, dtype=np.float32),
                high=np.full((self.tree.nodes.shape[0], 3), max_pos, dtype=np.float32),
                shape=(self.tree.nodes.shape[0], 3),  
                dtype=np.float32
            ),
            "on_fire_nodes": spaces.Box(
                low = 0,
                high = len(self.tree.nodes) - 1,
                shape = (len(self.tree.nodes),),
                dtype = np.int32
            ),
            "protected_nodes": spaces.Box(
                low = 0,
                high = len(self.tree.nodes) - 1,
                shape = (len(self.tree.nodes),),
                dtype = np.int32
            ),
            "firefighter_position": spaces.Box(
                low=np.array([min_pos, min_pos, min_pos], dtype=np.float32),
                high=np.array([max_pos, max_pos, max_pos], dtype=np.float32),
                shape=(3,),
                dtype=np.float32
            )
        })

    def _get_obs(self):
        """
        Convert the current state of the environment into an observation.
        Returns:
        - A dictionary containing the observation.
        """
        # Node positions
        node_positions = np.array(self.tree.nodes_positions)

        # Fire state    
        burned = self.fire_state.burned_nodes
        burning = self.fire_state.burning_nodes

        on_fire = burned.union(burning)
        protected = self.fire_state.protected_nodes

        return {
            "node_positions": node_positions,
            "on_fire_nodes": np.array(list(on_fire)),
            "protected_nodes": np.array(list(protected)),
            "firefighter_position": self.ff_position
        }
    
    def step(self, action):
        if not self.fire_state.burning_nodes: 
            print("No burning nodes, initializing fire state.")
            self.fire_state.set_burning_nodes([self.root])
    
        else:
            self.fire_state.propagate()
    
        obs = self._get_obs()
        reward = 0  
        done = False  
    
    
        return obs, reward, done,  {}

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.
        Parameters:
        - seed: Random seed for reproducibility.
        - options: Additional options for resetting the environment.
        
        Returns:
        - A tuple containing the initial observation and info.
        """
        super().reset(seed=seed)
        
        # Reset fire state
        self.fire_state = FireState(self.tree)

        # Reset firefighter position
        self.ff_position = np.random.uniform(low=self.action_space.low, high=self.action_space.high)

        return self._get_obs(), {}
    
    def validate_init_params(self, num_nodes, node_positions, adjacency_matrix):
        """
        Validates the initialization parameters for the environment.
        Raises:
        - ValueError if the parameters are inconsistent.
        """
                # Validate input combinations
        if num_nodes is not None and (node_positions is not None or adjacency_matrix is not None):
            raise ValueError("Cannot specify both num_nodes and node_positions/adjacency_matrix")
        
        if node_positions is not None and adjacency_matrix is None:
            raise ValueError("If node_positions is provided, adjacency_matrix must also be provided")
        
        if adjacency_matrix is not None and node_positions is None:
            raise ValueError("If adjacency_matrix is provided, node_positions must also be provided")
        
