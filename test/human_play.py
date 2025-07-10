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
    num_nodes = len(obs['nodes_positions'])

    while not done:
        print("Current observation:")
        print(f"Feasible:\n{obs['feasible_nodes']}")
        print(f"On fire nodes: {obs['on_fire_nodes']}")
        print(f"Protected nodes: {obs['protected_nodes']}")
        print("Firefighter position:", obs['firefighter_position'])
        print("Remaining time:", obs['firefighter_remaining_time'])
        print("Protecting:", obs['firefighter_protecting_node'])
        print("Which node do you want to protect?")
        nodes_distances = obs['feasible_nodes']
        for feasible in nodes_distances:
            print(f"Candidate {feasible[0]}: {feasible[1]:.2f} time to reach")
        res = input("Enter node index to protect (or 'exit' to quit): ")
        if res.lower() == 'exit':
            print("Exiting human play test.")
            break
        
        node_index = int(res)

        if 0 <= node_index < num_nodes:
            obs, reward, done, info = env.step(node_index)
            print(f"Protected node {node_index}.")
        else:
            print("Invalid node index. Please try again.")
            

    print("Final observation:")
    print(f"Feasible nodes: {obs['feasible_nodes']}")
    print("Firefighter position:", obs['firefighter_position'])
    print("Remaining time:", obs['firefighter_remaining_time'])
    print("Protecting:", obs['firefighter_protecting_node'])

test_human_play()