import sys
import os
from pathlib import Path

# Añadir la carpeta src al path de Python
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environment import FFProblemEnv

def test_human_play():
    env = FFProblemEnv(
        space_limits=(-1, 1),
        speed=1.0,
        num_nodes=5,
        root_degree=2,
        type_root_degree='exact'
    )
    obs = env.reset()
    print("Initializing fire...")
    obs, reward, done, info = env.step(None)  # Start the fire propagation

    while not done:
        print("Current observation:")
        print(f"On fire nodes: {obs['on_fire_nodes']}")
        print("Firefighter position:", obs['firefighter_position'])
        print("Remaining time:", obs['firefighter_remaining_time'])
        print("Protectig:", obs['firefighter_protecting_node'])
        print("Which node do you want to protect?")
        nodes_positions = obs['nodes_positions']
        for i, pos in enumerate(nodes_positions):
            print(f"Node {i} position: {pos}")
        res = input("Enter node index to protect (or 'exit' to quit): ")
        if res.lower() == 'exit':
            print("Exiting human play test.")
            break
        
        node_index = int(res)
        if 0 <= node_index < len(nodes_positions):
            obs, reward, done, info = env.step(node_index)
            print(f"Protected node {node_index}.")
        else:
            print("Invalid node index. Please try again.")
            

test_human_play()