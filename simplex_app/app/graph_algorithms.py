import heapq # Pour la file de priorité de Dijkstra et Prim
from collections import defaultdict, deque

# --- Fonctions Utilitaires pour les Graphes ---

def parse_adjacency_matrix(num_nodes, node_names, adj_matrix_dict, is_directed=False):
    """
    Transforme la matrice d'adjacence (dict {"i-j": val}) en :
    - adj: dict {node_name: [(voisin, poids), ...]}
    - nodes_set: set des noms de nœuds
    - edges_for_mst_and_viz: liste (u, v, poids)
    """
    adj = {name: [] for name in node_names}
    nodes_set = set(node_names)
    edges = []
    for i in range(num_nodes):
        u = node_names[i]
        for j in range(num_nodes):
            v = node_names[j]
            key = f"{i}-{j}"
            val = adj_matrix_dict.get(key, "0")
            try:
                weight = float(val)
            except (ValueError, TypeError):
                continue
            if weight and weight != 0:
                adj[u].append((v, weight))
                if is_directed or i < j:
                    edges.append((u, v, weight))
    return adj, nodes_set, edges

def parse_graph_data(graph_data_str, is_directed=False):
    """
    Parse les données du graphe depuis une chaîne de caractères.
    Format attendu par ligne : "node1 node2 [weight]" (weight est optionnel, défaut à 1).
    Retourne un dictionnaire représentant la liste d'adjacence et un ensemble de tous les nœuds.
    Pour les graphes non dirigés, ajoute l'arête inverse.
    Exemple de retour pour la liste d'adjacence:
    {
        'A': [('B', 5), ('C', 2)],
        'B': [('A', 5), ('C', 1)],  # Si non-dirigé
        'C': [('A', 2), ('B', 1), ('D', 4)], # Si non-dirigé
        'D': [('C', 4)] # Si non-dirigé
    }
    """
    adj = defaultdict(list)
    nodes = set()
    edges_for_mst = [] # Liste d'arêtes (u, v, poids) pour Prim/Kruskal

    lines = graph_data_str.strip().split('\n')
    for line_num, line in enumerate(lines):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 2:
            raise ValueError(f"Ligne {line_num+1} invalide: '{line}'. Format attendu: node1 node2 [poids]")
        
        u, v = parts[0], parts[1]
        weight = 1.0 # Poids par défaut
        if len(parts) > 2:
            try:
                weight = float(parts[2])
            except ValueError:
                raise ValueError(f"Ligne {line_num+1}: Poids invalide '{parts[2]}' pour l'arête {u}-{v}.")

        nodes.add(u)
        nodes.add(v)
        adj[u].append((v, weight))
        edges_for_mst.append((u, v, weight)) # Toujours ajouter pour MST, on le traitera comme non-dirigé là-bas

        if not is_directed:
            adj[v].append((u, weight))
            # Pour MST, si non dirigé, on ne veut pas dupliquer l'arête.
            # edges_for_mst contiendra u-v, on n'ajoute pas v-u explicitement ici,
            # mais Kruskal et Prim devront en tenir compte.
            # Prim gère bien avec la liste d'adjacence symétrique.
            # Kruskal utilisera la liste edges_for_mst.

    return adj, nodes, edges_for_mst


# --- Algorithme de Dijkstra ---
def dijkstra(adj, start_node, end_node=None):
    """
    Implémentation de l'algorithme de Dijkstra.
    adj: Liste d'adjacence du graphe.
    start_node: Nœud de départ.
    end_node: Nœud d'arrivée (optionnel). Si None, calcule les distances vers tous les nœuds.
    Retourne: (distances, predecessors)
              distances: dict {node: distance_from_start}
              predecessors: dict {node: previous_node_in_shortest_path}
              Si end_node est spécifié, retourne aussi le chemin et la distance pour ce nœud.
    """
    if start_node not in adj and start_node not in [item for sublist in adj.values() for item, _ in sublist]:
        # Check if start_node exists at all (even if it has no outgoing edges)
        all_nodes_in_graph = set(adj.keys())
        for u_node in adj:
            for v_node, _ in adj[u_node]:
                all_nodes_in_graph.add(v_node)
        if start_node not in all_nodes_in_graph:
            raise ValueError(f"Nœud de départ '{start_node}' non trouvé dans le graphe.")


    distances = {node: float('inf') for node in adj}
    # S'assurer que tous les nœuds (y compris ceux sans arêtes sortantes mais présents comme destination) sont dans distances
    all_graph_nodes = set(adj.keys())
    for u_node in adj:
        for v_node, _ in adj[u_node]:
            all_graph_nodes.add(v_node)
    for node in all_graph_nodes:
        if node not in distances:
            distances[node] = float('inf')

    distances[start_node] = 0
    predecessors = {node: None for node in all_graph_nodes}
    
    # File de priorité : (distance, node)
    pq = [(0, start_node)]
    
    processed_nodes = set()

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        # Ignorer si on a déjà trouvé un chemin plus court (ou si déjà traité pour certains algos)
        if current_node in processed_nodes and current_distance > distances[current_node]: # ou juste if current_node in processed_nodes
            continue
        processed_nodes.add(current_node)

        # Si on a atteint le nœud d'arrivée (optimisation)
        if end_node and current_node == end_node:
            break # On a trouvé le chemin le plus court vers la destination

        for neighbor, weight in adj.get(current_node, []): # adj.get pour gérer nœuds sans arêtes sortantes
            if weight < 0:
                raise ValueError("Dijkstra ne fonctionne pas avec des poids d'arêtes négatifs.")
            distance_through_current = current_distance + weight
            if distance_through_current < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance_through_current
                predecessors[neighbor] = current_node
                heapq.heappush(pq, (distance_through_current, neighbor))
    
    if end_node:
        path = []
        current = end_node
        if distances.get(current, float('inf')) == float('inf'): # Pas de chemin trouvé
            return float('inf'), [], distances, predecessors # Distance, path, all_distances, all_predecessors
        
        while current:
            path.append(current)
            current = predecessors[current]
            if current and current not in predecessors: # Sécurité
                 break
        path.reverse()
        return distances[end_node], path, distances, predecessors
    
    return None, None, distances, predecessors # Si pas de end_node spécifique


# --- Algorithme de Prim (Arbre Couvrant Minimum - MST) ---
def prim_mst(adj, nodes, start_node=None):
    """
    Implémentation de l'algorithme de Prim pour trouver l'Arbre Couvrant Minimum.
    Le graphe est traité comme non-dirigé.
    adj: Liste d'adjacence.
    nodes: Ensemble de tous les nœuds.
    start_node: Nœud de départ optionnel. Si None, prend le premier nœud trouvé.
    Retourne: (total_weight, mst_edges)
    """
    if not nodes:
        return 0, []
    
    if start_node is None or start_node not in nodes:
        start_node = next(iter(nodes)) # Prend un nœud arbitraire pour commencer si non spécifié ou invalide

    mst_edges = []
    total_weight = 0
    visited = {start_node}
    # File de priorité: (weight, node_from, node_to)
    edges_heap = []

    def add_edges_from_node(node):
        for neighbor, weight in adj.get(node, []):
            if neighbor not in visited:
                heapq.heappush(edges_heap, (weight, node, neighbor))

    add_edges_from_node(start_node)

    while edges_heap and len(visited) < len(nodes):
        weight, u, v = heapq.heappop(edges_heap)
        
        if v not in visited:
            visited.add(v)
            mst_edges.append((u, v, weight))
            total_weight += weight
            add_edges_from_node(v)
            
    if len(visited) != len(nodes) and len(nodes) > 1:
        print(f"AVERTISSEMENT Prim: Graphe non connexe. MST partiel trouvé pour la composante de {start_node}.")
        # On retourne le MST de la composante connexe trouvée.

    return total_weight, mst_edges

# --- Algorithme de Kruskal (Arbre Couvrant Minimum - MST) ---
class DisjointSetUnion: # Structure DSU (Union-Find) pour Kruskal
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i]) # Path compression
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Union by rank
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False

def kruskal_mst(nodes, edges_list):
    """
    Implémentation de l'algorithme de Kruskal pour trouver l'Arbre Couvrant Minimum.
    Le graphe est traité comme non-dirigé.
    nodes: Ensemble de tous les nœuds.
    edges_list: Liste de tuples (u, v, weight).
    Retourne: (total_weight, mst_edges)
    """
    if not nodes or not edges_list:
        return 0, []

    mst_edges = []
    total_weight = 0
    
    # Trier les arêtes par poids
    sorted_edges = sorted(edges_list, key=lambda item: item[2])
    
    dsu = DisjointSetUnion(nodes)
    
    num_edges_in_mst = 0
    for u, v, weight in sorted_edges:
        if dsu.union(u, v): # Si u et v ne sont pas déjà dans le même ensemble
            mst_edges.append((u, v, weight))
            total_weight += weight
            num_edges_in_mst += 1
            if num_edges_in_mst == len(nodes) - 1: # MST complet trouvé
                break
                
    if num_edges_in_mst < len(nodes) - 1 and len(nodes) > 1:
         print(f"AVERTISSEMENT Kruskal: Graphe non connexe. MST partiel trouvé.")

    return total_weight, mst_edges


# --- Algorithme de Ford-Fulkerson (Flux Maximum) ---
def ford_fulkerson_bfs(graph, s, t, parent):
    """Trouve un chemin augmentant de s à t en utilisant BFS."""
    visited = {node: False for node in graph}
    queue = deque()
    
    queue.append(s)
    visited[s] = True
    parent[s] = -1 # Marqueur pour la source
    
    while queue:
        u = queue.popleft()
        for v, capacity in graph[u].items(): # graph[u] est un dict {neighbor: residual_capacity}
            if not visited[v] and capacity > 0:
                queue.append(v)
                visited[v] = True
                parent[v] = u
                if v == t:
                    return True # Chemin trouvé
    return False # Pas de chemin augmentant trouvé

def ford_fulkerson(adj_list_with_capacities, nodes_set, source, sink):
    """
    Implémentation de l'algorithme de Ford-Fulkerson (avec Edmonds-Karp via BFS).
    adj_list_with_capacities: Liste d'adjacence où les poids sont les capacités.
                              Le graphe est traité comme dirigé.
    nodes_set: Ensemble de tous les nœuds.
    source: Nœud source.
    sink: Nœud puits.
    Retourne: La valeur du flux maximum.
    """
    if source not in nodes_set or sink not in nodes_set:
        raise ValueError("Source ou puits non trouvé dans le graphe.")
    if source == sink:
        return 0

    # Créer le graphe résiduel à partir de la liste d'adjacence
    # Le graphe résiduel stocke les capacités résiduelles.
    # Il est représenté par un dictionnaire de dictionnaires: graph[u][v] = capacité résiduelle de u à v
    residual_graph = defaultdict(lambda: defaultdict(float))
    all_nodes = set(nodes_set) # S'assurer que tous les nœuds sont présents

    for u, neighbors in adj_list_with_capacities.items():
        for v, capacity in neighbors:
            if capacity < 0:
                raise ValueError("Ford-Fulkerson ne gère pas les capacités négatives.")
            residual_graph[u][v] = capacity
            all_nodes.add(u)
            all_nodes.add(v)
            # Initialiser les arêtes inverses à 0 si elles n'existent pas (nécessaire pour le résiduel)
            if u not in residual_graph[v]:
                 residual_graph[v][u] = 0.0


    parent = {node: None for node in all_nodes} # Pour stocker le chemin augmentant
    max_flow = 0.0

    # Tant qu'il existe un chemin augmentant de la source au puits dans le graphe résiduel
    while ford_fulkerson_bfs(residual_graph, source, sink, parent):
        path_flow = float('Inf')
        # Trouver la capacité résiduelle minimum (flux de chemin) du chemin trouvé
        s_curr = sink
        while s_curr != source:
            u_prev = parent[s_curr]
            path_flow = min(path_flow, residual_graph[u_prev][s_curr])
            s_curr = u_prev
        
        # Mettre à jour les capacités résiduelles des arêtes du chemin
        # et des arêtes inverses
        v_curr = sink
        while v_curr != source:
            u_prev = parent[v_curr]
            residual_graph[u_prev][v_curr] -= path_flow
            residual_graph[v_curr][u_prev] += path_flow # Arête inverse
            v_curr = u_prev
            
        # Ajouter le flux de chemin au flux maximum global
        max_flow += path_flow
        
    return max_flow

# --- Vous pouvez ajouter d'autres algorithmes ici ---
