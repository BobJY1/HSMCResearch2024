class Graph(object):
  # Initialize the matrix
  def __init__(self, size):
    self.adjMatrix = [[0 for i in range(size)] for i in range(size)]
    for i in range(size):
      for j in range(size):
        self.adjMatrix[i][j] = 0
    self.size = size
    self.cliques = []

  # Add edges
  def add_edge(self, v1, v2):
    if v1 == v2:
      print("Same vertex %d and %d" % (v1, v2))
    self.adjMatrix[v1][v2] = 1
    self.adjMatrix[v2][v1] = 1
    return self

  # Remove edges
  def remove_edge(self, v1, v2):
    if self.adjMatrix[v1][v2] == 0:
      print("No edge between %d and %d" % (v1, v2))
      return
    self.adjMatrix[v1][v2] = 0
    self.adjMatrix[v2][v1] = 0
    return self

  #Add node (NOT vertex duplication)
  def add_node(self):
    for i in range(len(self.adjMatrix)):
      self.adjMatrix[i].append(0)
    self.size += 1
    self.adjMatrix.append([0 for i in range(self.size)])
    return self

  def __len__(self):
    return self.size

  # Print the matrix
  def print_matrix(self):
    for row in self.adjMatrix:
      print(row)

  def return_complement(self):
    h = Graph(self.size)
    complement_graph = [[(1 - x) for x in self.adjMatrix[j]]
                        for j in range(self.size)]
    for i in range(self.size):
      complement_graph[i][i] = 0
    h.adjMatrix = complement_graph
    return h

  def append_clique(self, clique):
    if clique not in self.cliques:
      self.cliques.append(clique)

  def remove_vertex(self, v):
    h = Graph(self.size - 1)
    h.adjMatrix = []
    for u in range(self.size):
      if u != v:
        h.adjMatrix.append(self.adjMatrix[u][:v] + self.adjMatrix[u][v + 1:])
    return h

  def edge_list(self):
    edges = []
    for u in range(self.size):
      for v in range(u + 1, self.size):
        if (self.adjMatrix[u][v] == 1):
          edges.append((u, v))
    return edges
  def remove(self, *v):
    removed = 0
    graph = self
    for i in sorted(v):
      graph = graph.remove_vertex(i - removed)
      removed += 1
    self.size = graph.size
    self.adjMatrix = graph.adjMatrix
    return self
