import sys
import os
from pathlib import Path

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
        feasible_nodes = obs['feasible_nodes']
        if not feasible_nodes:
            print("No feasible nodes to protect. No action can be taken.")
            res = None
        else:
            for feasible in feasible_nodes:
                print(f"Candidate {feasible[0]}: {feasible[1]:.2f} time to reach")
            res = input("Enter node index to protect: ")
            res = int(res)
        
        if(res != None):
            if 0 <= res < num_nodes:
                print(f"Protected node {res}.")
            else:
                print("Invalid node index. Please try again.")
        
        obs, reward, done, info = env.step(res)


    print("Final observation:")
    print(f"Feasible nodes: {obs['feasible_nodes']}")
    print(f"On fire nodes: {obs['on_fire_nodes']}")
    print(f"Protected nodes: {obs['protected_nodes']}")
    print("Firefighter position:", obs['firefighter_position'])
    print("Remaining time:", obs['firefighter_remaining_time'])
    print("Protecting:", obs['firefighter_protecting_node'])

test_human_play()