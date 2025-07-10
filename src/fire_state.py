class FireState:
    def __init__(self, tree):
        self.tree = tree
        self.burned_nodes = set()  
        self.burning_nodes = set()
        self.protected_nodes = set()
        self.feasible_nodes = {}
    
    def set_burned_nodes(self, burned_nodes):
        self.burned_nodes = burned_nodes

    def set_burning_nodes(self, burning_nodes):
        self.burning_nodes = burning_nodes

    def set_feasible_nodes(self, feasible_nodes):
        self.feasible_nodes = feasible_nodes

    def propagate(self):
        new_burning_nodes = set()
        
        for node in self.burning_nodes:
            neighbors = self.tree.get_neighbors(node)  # Method in the Tree class to get neighboring nodes
            for neighbor in neighbors:
                if neighbor not in self.burned_nodes and neighbor not in self.burning_nodes:
                    if neighbor not in self.protected_nodes:    # If node is not defended, it will burn
                        new_burning_nodes.add(neighbor)
        
        # Update the state of the nodes
        self.burned_nodes.update(self.burning_nodes)
        self.set_burning_nodes(new_burning_nodes)

    def is_completely_burned(self):
        """
        Checa si ya no hay nodos por quemar
        """
        if not self.burning_nodes and not self.burned_nodes:
            return False

        for node in self.burning_nodes:
            neighbors = self.tree.get_neighbors(node)
            for neighbor in neighbors:
                if neighbor not in self.burned_nodes and neighbor not in self.burning_nodes and neighbor not in self.protected_nodes:
                    return False
        return True

    def print_info(self):
        """
        Muestra el estado del fuego
        """
        print(f"Burned Nodes: {self.burned_nodes}")
        print(f"Burning Nodes: {self.burning_nodes}")
        print(f"Protected Nodes: {self.protected_nodes}")