import sys
import os
from pathlib import Path

# Añadir la carpeta src al path de Python
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environment import FFProblemEnv
import numpy as np

SPEED = 0.1  # Velocidad del bombero

def print_observation(obs):
    """
    Imprime la observación del entorno.
    """
    print("Observation:")
    for key, value in obs.items():
        print(f"{key}: {value.shape} with values: {value}")
    print("\n")

def test_tree_creation():
    space_limits = (-1.0, 1.0) 
    num_nodes = 10  
    root_degree = 3
    type_root_degree = 'min'
    
    # Caso 1: Crear entorno con árbol aleatorio
    print("=== Probando creación con árbol aleatorio ===")
    try:
        env_random = FFProblemEnv(
            space_limits=space_limits,
            speed=SPEED,
            num_nodes=num_nodes,
            root_degree=root_degree,
            type_root_degree=type_root_degree
        )

        print("✔ Éxito: Entorno creado con árbol aleatorio")
        
    except Exception as e:
        print(f"✖ Error al crear entorno con árbol aleatorio: {str(e)}")
        return

    # Caso 2: Crear entorno con árbol predefinido
    print("\n=== Probando creación con árbol predefinido ===")
    try:
        node_positions = np.random.uniform(low=-1, high=1, size=(5, 3))  # 5 nodos aleatorios en 3D
        adjacency_matrix = np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0]
        ])
        
        env_predefined = FFProblemEnv(
            space_limits=space_limits,
            speed=SPEED,
            node_positions=node_positions,
            adjacency_matrix=adjacency_matrix
        )
        print("✔ Éxito: Entorno creado con árbol predefinido")
    except Exception as e:
        print(f"✖ Error al crear entorno con árbol predefinido: {str(e)}")
        return
    
def test_initial_reset():
    try:
        space_limits = (-1.0, 1.0)
        num_nodes = 10
        root_degree = 3
        type_root_degree = 'min'
        
        env = FFProblemEnv(
            space_limits=space_limits,
            speed=SPEED,
            num_nodes=num_nodes,
            root_degree=root_degree,
            type_root_degree=type_root_degree
        )
        
        obs, inf = env.reset()
        
        print("=== Probando reset inicial del entorno para árbol aleatorio ===")

        print_observation(obs)

        print("✔ Éxito: Reset inicial del entorno realizado correctamente")

    except Exception as e:
        print(f"✖ Error al realizar el reset inicial del entorno: {str(e)}")
        return

def test_initial_reset_predefined():
    try:
        space_limits = (-1.0, 1.0)
        node_positions = np.random.uniform(low=-1, high=1, size=(5, 3))  # 5 nodos aleatorios en 3D
        adjacency_matrix = np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0]
        ])

        env = FFProblemEnv(
            space_limits=space_limits,
            speed=SPEED,
            node_positions=node_positions,
            adjacency_matrix=adjacency_matrix
        )

        obs, inf= env.reset()

        print("=== Probando reset inicial del entorno para árbol predefinido ===")
        print_observation(obs)

        print("✔ Éxito: Reset inicial del entorno predefinido realizado correctamente")

    except Exception as e:
        print(f"✖ Error al realizar el reset inicial del entorno predefinido: {str(e)}")
        return

def test_step():
    space_limits = (-1.0, 1.0)
    num_nodes = 10
    root_degree = 3
    type_root_degree = 'min'
        
    env = FFProblemEnv(
        space_limits=space_limits,
        speed=SPEED,
        num_nodes=num_nodes,
        root_degree=root_degree,
        type_root_degree=type_root_degree
    )
    obs = env.reset()
        
    obs, reward, done, info = env.step(1)
    
    print("=== Probando paso del entorno ===")
    print_observation(obs)

        # test other action
    obs, reward, done, info = env.step(1)
    print("=== Probando otro paso del entorno ===")
    print_observation(obs)
    print("✔ Éxito: Paso del entorno realizado correctamente")

def test_init():
    try:
        space_limits = (-1.0, 1.0)
        num_nodes = 10
        root_degree = 3
        type_root_degree = 'min'

        env = FFProblemEnv(
            space_limits=space_limits,
            speed=SPEED,
            num_nodes=num_nodes,
            root_degree=root_degree,
            type_root_degree=type_root_degree
        )

        print("=== Probando inicialización del entorno para árbol aleatorio ===")
        print('Firefighter initial position:', env.ff.position)
        print("✔ Éxito: Inicialización del entorno realizada correctamente")

    except Exception as e:
        print(f"✖ Error al realizar la inicialización del entorno: {str(e)}")
        return


if __name__ == "__main__":
    test_tree_creation()
    test_initial_reset()
    test_initial_reset_predefined()
    test_step()
    test_init()