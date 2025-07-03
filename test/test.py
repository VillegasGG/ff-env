import sys
import os
from pathlib import Path

# Añadir la carpeta src al path de Python
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environment import FFProblemEnv
import numpy as np

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
            node_positions=node_positions,
            adjacency_matrix=adjacency_matrix
        )
        print("✔ Éxito: Entorno creado con árbol predefinido")
    except Exception as e:
        print(f"✖ Error al crear entorno con árbol predefinido: {str(e)}")
        return

if __name__ == "__main__":
    test_tree_creation()