import numpy as np
import time
import networkx as nx
import random
from sporadics import m1
import sat_colorability_solver
from graph import Graph
import sysconfig
import daft
import itertools
import os
import math

#print(sysconfig.get_paths()["purelib"])


def check_triangle_free(g):
  adj = np.array(g.adjMatrix)
  squ = np.matmul(adj, adj)
  tri = np.matmul(squ, adj)
  trace = np.trace(tri)
  if trace == 0:
    return True
  else:
    return False


#Geeks for geeks
def check_two_colorable(adjMatrix):
  for x in range(len(adjMatrix)):
    colorArr = [-1] * len(adjMatrix)
    colorArr[x] = 1
    queue = []
    queue.append(x)
    while queue:
      u = queue.pop()
      if adjMatrix[u][u] == 1:
        return False
      for v in range(len(adjMatrix)):
        if adjMatrix[u][v] == 1 and colorArr[v] == -1:
          colorArr[v] = 1 - colorArr[u]
          queue.append(v)
        elif adjMatrix[u][v] == 1 and colorArr[v] == colorArr[u]:
          return False
  return True


#get list of neighbors of a vertex
def N(vertex, adjMatrix):
  c = 0
  l = []
  for i in adjMatrix[vertex]:
    if i == 1:
      l.append(c)
    c += 1
  return l

#some algorithm for clique testing
def bronkerbosch(r, p, x, g):
  if len(p) == 0 and len(x) == 0:
    g.append_clique(r)
  for vertex in p[:]:
    r_new = r[:]
    r_new.append(vertex)
    p_new = [val for val in p if val in N(vertex, g.adjMatrix)]
    x_new = [val for val in x if val in N(vertex, g.adjMatrix)]
    bronkerbosch(r_new, p_new, x_new, g)
    p.remove(vertex)
    x.append(vertex)
  g.cliques.sort(key=len)
  return g

#helper for next func
def check_mpg_complement_inner(g, fast=False, silent=False):
  if not silent:
    print("Valid coloring list:")
  colorings = sat_colorability_solver.allSolutions(g.size, g.edge_list())
  if colorings == []:
    return None # this means no coloring found

  edge_add = []
  for coloring in colorings:
    if (not silent): print(coloring)
    while (edge := check_can_add(g, coloring, fast=True, silent=silent)) != (-1, -1):
      edge_add += [edge]
      if fast: 
        return edge_add
    
  return edge_add

#check if graph is mpgc
def check_mpg_complement(g, fast=False, silent=False):
  t1 = time.time()
  if not is_complement_connected(g):
    if not silent:
      print("MPG is not connected.")
    return False
  if check_triangle_free(g):
    minimal = check_mpg_complement_inner(g, fast=fast, silent=silent)
    

    t2 = time.time()
    if not silent:
      print("Time taken:", str(t2 - t1), "seconds")
    if (minimal == None):
      if not silent:
        print("Graph is not colorable")
      return False
    if (minimal == []):
      if not silent:
        print("Graph is minimal")
      return True
    else:
      if not silent:
        print("Not minimal")
      return False
  if not silent:
    print("Not triangle free")
  t2 = time.time()
  if not silent:
    print("Time taken:", str(t2 - t1), "seconds")
  return False

#find if edge can be added given graph and coloring. returns edge if possible, otherwise (-1, -1)
def check_can_add(g, c, fast=False, silent=False, permutation=None):
  if permutation == None:
    permutation = range(g.size)
  can_add = (-1, -1)
  for k in range(len(permutation)):
    u = permutation[k]
    for l in range(k+1,len(permutation)):
      v = permutation[l]
      if (c[u] != c[v]) and (g.adjMatrix[u][v] == 0):
        g.adjMatrix[u][v] = 1
        if check_triangle_free(g):
          if not silent:
            print("Added edge between nodes", str(u), "and", str(v))
          can_add = (u, v)
          if (fast): return can_add
        g.adjMatrix[u][v] = 0
  return can_add


#input "g" is the complement of a graph
#tells if g represents the complement of a superminimal prime graph
#Need to consider graphs of size <=2 ???
#Also requires graph to be connected
def check_superminimal_complement_helper(g, domain=None, original_coloring=None, fast=False, silent=False):
  removable = []
  if (domain == None):
    domain = range(g.size)
  for v in domain:
    if not silent:
      print("---------------- Removing vertex", str(v))
    h = g.remove_vertex(v)
    
    if (original_coloring != None):
      if not check_mpg_complement(
          h,
          fixed_coloring={i:(original_coloring[:v] + original_coloring[v + 1:])[i] for i in range(g.size - 1)},
          fast=fast, silent=silent):
        continue
    if check_mpg_complement(h, fast=fast, silent=silent):
      if not silent:
        print("Vertex", v, "was removed!")
      removable.append(v)
      if (fast):
        return False
  if not silent:
    print("----------------")
  if (len(removable) == 0):
    if not silent:
      print("Graph is superminimal!")
    return True
  else:
    if not silent:
      print("Graph is not minimal!")
      print(f"Can remove {removable}.")
    return False

#check if graph is superminimal complement lol
def check_superminimal_complement(g, fast=False, silent=False):
  if check_mpg_complement(g, fast=fast, silent=silent):
    return check_superminimal_complement_helper(g, fast=fast, silent=silent)
  return False

#check if coloring is valid on graph g, with edge list e. returns first failing edge or (-1, -1) if successful
def check_coloring(g, c, e):
  for (u, v) in e:  #ordered from smallest to largest
    if (c[u] == c[v]): return (u, v)
  return (-1, -1)

#compare to graphs to see if they are isomorphic
def compgraph(g, h):
  g1 = nx.Graph()
  g1.add_edges_from(g.edge_list())
  h1 = nx.Graph()
  h1.add_edges_from(h.edge_list())
  return nx.is_isomorphic(g1, h1)

#check if complement is connected 
def is_complement_connected(g):
  g1 = nx.Graph()
  g1.add_nodes_from(range(g.size))
  for i in range(g.size):
    for j in range(i + 1, g.size):
      if (g.adjMatrix[i][j] == 0):
        g1.add_edge(i, j)
  return nx.is_connected(g1)

#generate
def generateCycle(n):
  g = Graph(n)
  g.add_edge(0, n - 1)
  for i in range(n-1):
    g.add_edge(i, i + 1)
  return g
#generate N graph, predecessor to W graphs, definitely fixed coloring mpgc but not necessarily superminimal
def generateNgraph(n):
  if (n % 3 != 0):
    raise Exception()
  n //= 3
  edge_list = []
  rows = [range(0, n), range(n, 2 * n), range(2 * n, 3 * n)]
  for row in range(2):
    for i in range(n):
      if (i % 2 == 0):
        for j in range(n):
          if (i - j <= 1 and j - i <= 1):
            edge_list.append((rows[row][i], rows[row + 1][j]))
      else:
        edge_list.append((rows[row][i], rows[row + 1][i]))
  for i in range(n):
    if (i % 2 == 0):
      for j in range(n):
        if (i - j < -1 or j - i < -1):
          edge_list.append((rows[0][i], rows[2][j]))
    else: 
      for j in range(n):
        if (i != j):
          edge_list.append((rows[0][i], rows[2][j]))
  g = Graph(3 * n)
  for edge in edge_list:
    g.add_edge(edge[0], edge[1])
  return g

#'''
#  2   6   10    14
#  1 3 5 7  9 11 
#0   4   8    12
#'''
#generate W graph as defined by me, proven to be superminimal, unique coloring
def generateWgraph(n): #n is multiple of 2
  if (n % 2 != 0): raise Exception()
  edge_list = []
  low = [i for i in range(n + 1) if (i % 4 == 0)]
  mid = [i for i in range(n - 1) if (i % 2 == 1)]
  high  = [i for i in range(n + 1) if (i % 4 == 2)]

  for i in mid:
    edge_list.append((i, i - 1))
    edge_list.append((i, i + 1))
    edge_list.append((i, i + 3))
  for i in high:
    for j in low:
      if (i - j != 2 and j - i != 2):
        edge_list.append((i, j))
  g = Graph(n + 1)
  for (v1, v2) in edge_list:
    g.add_edge(v1, v2)
  return g.remove(n - 1)

#generate g graph as defined by other paper, proven to be superminimal
def generateGgraph(n):
  if(n%6 != 0 and n%6 != 5): raise Exception()
  k = (n+2)//6
  edge_list = []
  for i in range(n):
    for l in range(k, 2*k):
      edge_list.append([i, (i + l) % n])
  g = Graph(n)
  for (v1, v2) in edge_list:
    g.add_edge(v1, v2)
  return g

#generate P graph, i forgot what this one is tbh but it might have some superminimals idk
def generatePgraph(n):
  edge_list = []
  if(n%3 == 2): 
    for i in range(n):
      for j in range(i + 1, n):
        if ((j - i) % 3 == 1):
          edge_list.append([i, j])
  else:
    raise Exception()
  g = Graph(n)
  for (v1, v2) in edge_list:
    g.add_edge(v1, v2)
  return g


#generate O graph, not superminimal family but has some superminimal components
def generateOgraph(n):
  if n % 4 != 2:
    raise Exception()
  n //= 2
  i = Graph(2*n)
  [i.add_edge(j, (j+1)%n) for j in range(n)]
  [i.add_edge(j+n, (j+1)%n+n) for j in range(n)]
  for k in range(0, n - 2, 2):
    [i.add_edge(j+n, (j+k)%n) for j in range(n)]
  return i
  
#generate a ship graph with n vertices, conjectured to always be superminimal
def generateShipgraph(n):
  if n % 3 == 1:
    raise Exception()

  k = (n - 2)//3
  if n % 3 == 0:
    g = generateShipgraph(n - 1)
    g.add_node()
    g.remove_edge(0, n - 2)
    g.add_edge(0, n - 1)
    g.add_edge(n - 2, n - 1)
    for i in range(k + 1, 2*k + 1):
      g.add_edge(i, n - 1)
    return g
  
  g = Graph(n)
  for i in range(2, 2*k):
    for j in range(k, 2*k):
      if (i + j < n - 2):
        g.add_edge(i, i + j)
  for i in range(n - k - 1, n):
    g.add_edge(0, i)
  for i in range(k + 1):
    g.add_edge(n - 1, i)
  g.add_edge(1, k+1)
  g.add_edge(1, k+2)
  g.add_edge(1, n-2)
  for i in range(k + 3, 2*k + 1):
    g.add_edge(n - 2, i)
  return g

#print matrix in good format for visulaization
def printm(matrix):
  for i in range(len(matrix)):
    [print("0" if matrix[i][j] == 0 else "1", end=" ") for j in range(len(matrix[i]))]
    print()

#generate mpgc, not necessarily accurate
def generateRandomMinimal(n):
  g = Graph(n)
  coloring = [random.randrange(0,3) for i in range(n)]
  addRandomMinimal(g, coloring)
  return [g, coloring]

#given a unique coloring, add arbitrary edges to graph until it becomes a mpgc (likely not superminimal)
def addRandomMinimal(g, c):
  nodes = list(range(g.size))
  random.shuffle(nodes)
  edge = check_can_add(g, c, fast=True, permutation=nodes, silent=True)
  while edge != (-1, -1):
    g.add_edge(*edge)
    random.shuffle(nodes)
    edge = check_can_add(g, c, fast=True, permutation=nodes, silent=True)
  return g

#util for permuting the order of vertices in a graph
def permute(g, perm):
  new_adj = [[0]*g.size for i in range(g.size)]
  for i in range(g.size):
    for j in range(g.size):
      new_adj[i][j] = g.adjMatrix[perm[i]][perm[j]]
  g.adjMatrix = new_adj

#check the conjecture that any 4-path in a mpgc is also part of a 5-cycle
def find5cycle(g, c):
  for (u, v) in g.edge_list():
      for w in range(g.size):
        if (g.adjMatrix[v][w] != 1 or w == u):
          continue
        for x in range(g.size):
          if (g.adjMatrix[w][x] != 1 or c[u] != c[x] or x in [u, v]):
            continue
          cycle = False
          for y in range(g.size):
            if (y in [u, v, w, x]):
              continue
            if (g.adjMatrix[x][y] == 1 and g.adjMatrix[y][u] == 1):
              cycle = True
          if (not cycle):
            print(u, v, w, x, "not part of a cycle!!!!")
            return True
  return False

#checks the conjecture that every edge in a mpgc is part of a 5-cycle
def find5cycle_strong(g):
  for (u, v) in g.edge_list():
      cycle = False
      for w in range(g.size):
        if (g.adjMatrix[v][w] != 1 or w == u):
          continue
        for x in range(g.size):
          if (g.adjMatrix[w][x] != 1 or x in [u, v]):
            continue
          for y in range(g.size):
            if (y in [u, v, w, x]):
              continue
            if (g.adjMatrix[x][y] == 1 and g.adjMatrix[y][u] == 1):
              cycle = True
      if (not cycle):
        print(u, v, "not part of a cycle!!!!")
        return True
  return False  

#use nx to find any induced n-cycles of g
def find_induced_n_cycle(g, n):
  g1 = nx.Graph()
  g1.add_edges_from(g.edge_list())
  h1 = nx.cycle_graph(n)
  gm = nx.isomorphism.GraphMatcher(g1, h1)
  #return gm.subgraph_is_isomorphic()
  print(gm.subgraph_is_isomorphic())
  return gm.subgraph_isomorphisms_iter()

#generate the brinkmann graph, a square and triangle free 4 chromatic graph
def brinkmann_graph():
  g = Graph(21)
  for i in range(7):
    g.add_edge(i, (i+3)%7)
    g.add_edge(i, i + 7)
    g.add_edge(i, (i + 2)%7 + 7)
    g.add_edge(i + 7, i + 14)
    g.add_edge((i + 1)%7 + 7, i + 14)
    g.add_edge(i + 14, (i + 2)%7 + 14)
  return g

#find k-neighborhood of a graph g
def kneighbors(g, k):
  return [N(i, g.adjMatrix) for i in range(g.size)] if k == 1 else [{j for n in x for j in kneighbors(g, 1)[n]} for x in kneighbors(g, k - 1)]

#generate actually guaranteed minimal prime graphs
def guaranteedMPG(n):
  [g, c] = generateRandomMinimal(n)
  if (c.count(0) > len(c)/2 or c.count(1) > len(c)/2 or c.count(2) > len(c)/2):
    return guaranteedMPG(n)
  colorings = sat_colorability_solver.allSolutions(g.size, g.edge_list())
  if (len(colorings) > 100):
    #print(len(colorings), " too long.")
    return guaranteedMPG(n)
  #print(len(colorings))
  for coloring in colorings:
    if (check_coloring(g, coloring, g.edge_list())):
      while (edge := check_can_add(g, coloring, fast=True, silent=True)) != (-1, -1):
        g.add_edge(*edge)
        #print(edge)

  if is_complement_connected(g):
    return g

  return guaranteedMPG(n)

#check proportion of graphs with properties
def proportionOfMPG(n, iters, pred, epoch=20):
  t = time.time()
  count = 0
  for i in iters:
    g = guaranteedMPG(n)
    if (pred(g)):
      count += 1

    if i % 20 == 0:
      print(f"{count} / {i} at {count/i} percent in {time.time()-t}")

def construct_from_adj(adj):
  g = Graph(len(adj))
  g.adjMatrix = adj
  return g

def construct_from_edge_list(edges):
  g = Graph(max({v for edge in edges for v in edge}) + 1)
  for e in edges:
    g.add_edge(*e)
  return g

def generateSUPERMINIMAL(v):
  g = guaranteedMPG(v)
  end = False
  while (not end):
    end = True
    
    r = list(range(g.size))
    random.shuffle(r)
    
    for v in r:
      g2 = Graph(g.size)
      g2.adjMatrix = g.adjMatrix
      if check_mpg_complement(g2.remove(v), fast=True, silent=True):
        g.remove(v)
        end = False
        break
        
  return g


def superminimal_maker(v):
  g = generateSUPERMINIMAL(v)
  if (not check_superminimal_complement(g, silent=True, fast=True)):
    print("ERROR\n-----------------------")
    print(g.adjMatrix)

  #all the graphs from my testing
  some_graphs = [
    #superminimal
    #5 vertices
    generateCycle(5),
    #9 vertices
    generateNgraph(9),
    #13 vertices
    generateWgraph(14).add_edge(0,2).remove(1),
    #14 vertices
    generateOgraph(14),
    #15 vertices
    generateWgraph(18).remove(1,2,3),
    #17 vertices
    generateWgraph(18).add_edge(0,2).remove(1),
    generateWgraph(20).remove(1,2,3),
    #18 vertices
    generateOgraph(18),

    #and unique coloring
    #12 vertices
    generateNgraph(12),
    #14 vertices
    generateNgraph(15).remove(1),
    construct_from_adj([[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1], 
     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0], 
     [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0], 
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1], 
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0], 
     [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1], 
     [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0], 
     [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
     [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0], 
     [1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
     [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
     [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0], 
     [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
     [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]),
    #15 vertices
    generateNgraph(18).remove(1,3,14),
    #16 vertices
    generateNgraph(18).remove(1, 16),
    #17 vertices
    generateNgraph(21).remove(1,3,5,16),
    generateNgraph(21).remove(1,3,16,18)
  ]

  #families
  superminimal_families = [
    generateShipgraph,
    generateWgraph,
    generatePgraph
  ]
  
  #no comment
  for m in m1:
    some_graphs += [construct_from_edge_list(m)]

  iso = False
  for i in range(len(some_graphs)):
    gr = some_graphs[i]
    if (gr.size == g.size):
      if (compgraph(gr, g)):
        #print(f"isomorphic to sporadic {i}")
        iso = True

  for i in range(len(superminimal_families)):
    f = superminimal_families[i]
    try:
      gr = f(g.size)
      if (compgraph(gr, g)):
        #print(f"isomorphic to family {i}")
        iso = True
        
    except: 
      continue

  if (not iso):
    print("NEW SUPERMINIMAL?")
    #printm(g.adjMatrix)
    #print(g.adjMatrix)
    return g
  
def looped_superminimal_finder():
  #b = b'\x80\x04\x95w\x0e\x00\x00\x00\x00\x00\x00K\x03(J\x83I\x17>\x8a\x05\x1c9F\xe5\x00\x8a\x05\x8b\x8b\xe1\xa6\x00J\x86\x82Je\x8a\x05\xdb\x1a\x8a\xb5\x00Jx\xbc\x9b$\x8a\x05\xf3\x8e\x7f\xba\x00Jf\xf2F\x16J\xb1{\x08\x1c\x8a\x05\xf9\x7ff\xdf\x00J\x04\xcf\xf8~\x8a\x052[o\xf6\x00J@\x0e\x01OJN\x83\x04jJ\xe17\x05\x08\x8a\x05\r<\xab\xf7\x00\x8a\x05\x84\xc1\x8e\x81\x00JJ\xc1\x04>JX.o9J\xec\xf6=4Jp\xf0\xca\x7f\x8a\x056\xe7s\xd7\x00J\x8f\xb5\x8bA\x8a\x05\xfb\x1d\x89\xc3\x00Jz\xa4\rhJ\x89>\xf0b\x8a\x05iN\xff\xdb\x00J!\x83\xb3GJ\xc6\xe3\x928J\x02\xa1\xcbfJ\xcf{y?J\xa0\x1d\xb6\x02\x8a\x05H\xd3\x14\xb8\x00\x8a\x05#z\xe2\xa5\x00\x8a\x05\xe5k\x1f\xc9\x00JK\x87\x01(J g\xe0UJ\xeb\tGv\x8a\x05.\xf3O\xce\x00J\xc8\xfev\x15\x8a\x05\xde\x93\xb1\xfb\x00JA\xe1\xcbVJ\xbf*\xe1v\x8a\x05\xd0\xe1F\xb4\x00J\x87%\x91\x11\x8a\x05<\xf2A\x93\x00J\xe4WO\x0e\x8a\x05\x87\xb4w\xda\x00\x8a\x056x\xbd\x8b\x00\x8a\x05\xfdIn\xc4\x00\x8a\x05\x86\xaaj\x99\x00\x8a\x05\x0e\xd4\xe1\xa8\x00J*-\\K\x8a\x05A\xe2\\\xd3\x00\x8a\x05lK\xe5\xc5\x00\x8a\x05\xe9\xc1\xbd\xf1\x00\x8a\x05\xe4\xa8\xee\x86\x00J\xe7VlUJ\xeeAJ\x10JU\xc0o\x02J[j\xad@\x8a\x05!\xd1c\xf8\x00\x8a\x05C{\xa5\xdc\x00J\xe1\xa5\x19_\x8a\x05o\xefZ\xf7\x00\x8a\x05K\x92G\x94\x00J9\xb3\r,J/)\xfc\nJ_-\x02@\x8a\x050\t$\xb1\x00J\xc3*u\rJr0\xb9r\x8a\x05\xa6\xca\xfd\xda\x00J\xd7\x19\xf2E\x8a\x05\xac\x18\xb6\x9c\x00J%i\xfc"J$\xb1\x05\nJ\x83\xa74A\x8a\x05r\xd8\x91\xbf\x00\x8a\x05\x0eIa\x93\x00J\x0b\xca\xc2\x0c\x8a\x05+\xf8\xcb\x97\x00J\xbdb\xb0dJ\x87\xfah\x0f\x8a\x05\xb3\xa6t\x8a\x00JC\x82Qy\x8a\x05\xa6\xee\xda\xb4\x00\x8a\x05^\xf5\xf5\xee\x00J\to]M\x8a\x05\xe4\xa4\xa9\x8b\x00\x8a\x05c\xd0\x9e\xc2\x00J\xe0\xa7\xa79\x8a\x05\xc7\xf9\xca\xa5\x00\x8a\x05]\x0e$\x88\x00J\x9f\xc2f\x05Jc"Xw\x8a\x05\xfa\x15\xf4\xa7\x00J\x80M\x97\x0fJ\t\xe8\xa8.JU\xe5Y^J\xa5\xb0Y$\x8a\x05o\\\xc0\xad\x00Jl\xc4\xa1\x0fJHA\xbdHJa\xd17h\x8a\x05_B<\xc9\x00J\x91<?5J\x89\xae\xfc6J\xf5\xa1\x9cM\x8a\x05\xa0W\x85\x99\x00\x8a\x054w\xa5\xfb\x00\x8a\x05\xe7\xa4\xfa\xf6\x00\x8a\x05\x9e,t\x8d\x00J\xfd\xf08}\x8a\x05\xb6v\xaf\xc3\x00J\xd0\xd0\xc4BJ7q\xb6\x08J\xe1\x95\xa2+JQKA=\x8a\x05im\xcc\xb1\x00J\x18;\xd9<\x8a\x05\xae\xfb\xa8\xc2\x00\x8a\x05{\xb1\xb8\x85\x00J\xfd\x18t\x7fJ\xe2\x0b\xdeR\x8a\x05f95\xdf\x00Ja0^wJb\xf0\x11\x12\x8a\x05\xf5\x82\x08\x93\x00J\xad\'AiJ\xc0y2eJC*L}J\x02\xcc "J\xba\xc2\x93WJ\xac7\x051\x8a\x05\xd7Y\x87\xcb\x00JW\xc5\xefBJ\x12f\x85B\x8a\x05\xfe\xf4m\xfd\x00J\xb9<R\x16JX\xa1\xa1\x1eJrR\x9a\x1a\x8a\x05\xe9\x81\xd0\xa0\x00\x8a\x05a\x19\x0b\x83\x00J\x17Y^9J\xacqhw\x8a\x05PZ\x00\xda\x00J\xc3\xa0F\x0c\x8a\x056L\x93\x89\x00\x8a\x05\xe9&s\xe1\x00\x8a\x05\xe3\xe0\xbc\xdc\x00\x8a\x05\xa73\x15\xde\x00JC\xec\xc3A\x8a\x05\xc4\x0b(\x85\x00J\x9daN1\x8a\x05\x9a\xa9\xed\x83\x00\x8a\x05:\xb3\x03\xdb\x00J\xad\xc2\xc36\x8a\x05\xd7 N\x93\x00J\x8c\x98\x12\x0f\x8a\x05r\x01\xc3\xe1\x00\x8a\x05\xfb/2\xd3\x00\x8a\x05\xc9\xe0z\xee\x00\x8a\x05\xe4\x1d\xf2\xa9\x00J\xc8\xefS~J9\xf9\x8f|\x8a\x05\x88\xfe\xe8\xc0\x00\x8a\x05bl(\xdf\x00\x8a\x05\x7f\x86Y\xf4\x00\x8a\x05\x8cc\x9f\xa9\x00J\xe4\xfd\xa1\x0eJ\x9e\xb7\xf3\x10\x8a\x05\xe5\xe2W\x8f\x00\x8a\x05\xa7\xaf\x96\xbd\x00\x8a\x05\x1b5!\xc3\x00\x8a\x05K:\xcf\xaa\x00J\xc3\x14\x8f\x18\x8a\x05\xd5\xda"\xed\x00JX\xa5\x06CJ\xe3\xc2\xe15\x8a\x05qA|\xfb\x00\x8a\x05\xbf\xdd\x0b\x8e\x00J\x9a?\x8e`\x8a\x05N\xbaE\xb1\x00\x8a\x05\x00\xa2\xfb\xad\x00\x8a\x05_\xe9\xb8\xc3\x00J`\xc2A\x10\x8a\x05\xc9\xff\x16\xd4\x00\x8a\x05\xa75\xe4\xf2\x00J\xdd\x10vZ\x8a\x05^r\x84\xc6\x00\x8a\x05\xcdvS\xda\x00\x8a\x05\xcb\x81\x82\xb7\x00\x8a\x05>\x8a<\xf3\x00J$\xe5\x9dG\x8a\x05$\x83y\xb4\x00J\xa3OZrJ\xee~\x88}J\xe8RA{\x8a\x05\x06\x98k\xd2\x00\x8a\x05\xe88\x97\xba\x00J\xafx\x94,J\x96.\xe17Jf\x0e\xd4\x05J\xd7\xf2\xc8\x0eJ\x96\x17DFJv\xc2\xa9&\x8a\x05"\\|\xb8\x00\x8a\x05\x87\xd2p\xbc\x00JD\xdf\xa6\x14J\xf9\x12\xfamJ\xcc\xd4Z`\x8a\x054\xd0\x8b\xf6\x00\x8a\x05\x088@\xc9\x00\x8a\x05\x0e\x8a\x86\xd2\x00J\xf9\xa7\xaf9J\xceu\xae\x04\x8a\x05\xc1d`\xe4\x00JW\xe8\xccg\x8a\x05\x86\xc7\x11\xf7\x00\x8a\x05d{\xc9\x8a\x00\x8a\x05\xad\xfa\xdb\xbd\x00JR\xb3\x8exJ\\d\xfc\x7fJ<\xde\x9fqJPb\xf3RJ\x17\xbc\xbbRJ=\xc1\xdcuJj\xe8\xe0|J:\x97\x06xJc\xf34}JR\xb8K^J\x0f\x16C%\x8a\x05\xee\xc7y\xab\x00J\x8830JJ\xc4\x11\x9bxJ\x93C\xc3yJ\xd4K\x82\x1c\x8a\x05\x07\x98\xa9\xa2\x00\x8a\x05\xcb~V\xbc\x00J\x7f8\xbd\x1dJ\x0f5\x99>\x8a\x05&\xae\x8e\xa0\x00J\x7f\xf2oi\x8a\x05\xf3_\xc8\xbe\x00\x8a\x05\xc4\x13\x99\xbc\x00J\xed\xb5P\x1fJ\xae\xf1\xfb\x08\x8a\x05M\'\x13\xfe\x00J\x05x!,\x8a\x05\x0eq\x81\xbc\x00Jwf\xfdz\x8a\x05\xe8\x06E\xca\x00\x8a\x05\x05\x909\x8a\x00\x8a\x05A\xfb\xfc\x90\x00J@$[`Jk\xd0Fo\x8a\x05\x8e\xf3\xff\x82\x00J\x88F\xc6aJ\x1c\x12\xc28\x8a\x05\xd6\x06j\x9f\x00J\x13\xb5\xb4u\x8a\x05\x81\xfa\xc6\xae\x00\x8a\x05F{G\x8d\x00JP\xfb\xdc(\x8a\x05fl_\xb4\x00JStyoJ\xebI\x0f\x0cJ\xa3X=3\x8a\x05`\xd1\x0f\x82\x00J\xad\x1e\x03XJ{\x06fc\x8a\x05\x8b\xf2\xaa\xee\x00\x8a\x05\x99\x7fh\xb7\x00J\xbf\xc2{M\x8a\x05\xc7\x0e0\xfb\x00J\x1c\xd1\x8e\x15JQ\xd2\xc8\x0f\x8a\x05Z\xbf\xe8\xf2\x00\x8a\x058n\x19\x9e\x00\x8a\x05\xac\xb6R\xdd\x00\x8a\x05\xa8[\x85\xbe\x00J\xf5f\x9e+\x8a\x05\x00\xb5F\xca\x00J\x94z$X\x8a\x05\x1a}\x1c\xcd\x00J\xed\xcc\x0e\x12Jf\xe2\x0e\x1dJ\x82Q\xcd\nJX\x7f\xda_\x8a\x05\xbcZ\x98\xd6\x00J\xc5l\xeb4JH\xf4\x8d/\x8a\x05!\xda\x01\xe8\x00JV\xd2\\\x1c\x8a\x05\xa2@E\xf2\x00\x8a\x05-*\xba\xa4\x00J5\xbeS\x18J\xdf\x95/\x1dJ6\x95\x8eZ\x8a\x05\xf5UT\xb4\x00\x8a\x05\xa7\xf53\x8b\x00\x8a\x05\x98\xf3\xca\x85\x00JI\xea\xb3*\x8a\x05\xa4.\x0b\xe3\x00J5\xac\'G\x8a\x05\xce\xd9\\\xb5\x00J\xcc\x19<TJNTc\x03\x8a\x05\xe5GW\xdf\x00J\tP\x8f\x08\x8a\x05\x9b\xd3\x03\xc1\x00J\x94;t3\x8a\x05E\x06\xf6\x86\x00JQ\x9c\x1e5J\x9c\x15\xc4N\x8a\x05\xc89\x94\xb5\x00\x8a\x05C\xc4\x90\xc9\x00J\xd9\x9c\xf4:\x8a\x05\x97\xf53\x80\x00J\x9e\xd8\xd4\x08JSs\xden\x8a\x05\xef\xae\x12\xfc\x00JNY\xb4;\x8a\x05T\x19\x86\xa7\x00J6\\8$\x8a\x05\x1ft\xe2\xf7\x00J\xce\xd2\x90\\\x8a\x05\xa4\x8f\x87\xb8\x00\x8a\x05/_\x0b\xca\x00J\x80\xf2g5Jks\xd88\x8a\x05\xc5\xa91\xe2\x00\x8a\x05\x89\x9d\x0c\xb0\x00J\xd1S\x07$\x8a\x05\x0eH%\xa7\x00J\x1b?/]\x8a\x05\x82\xde\x7f\x8e\x00J\x9e@\x93\x00J\x05\x86\x08d\x8a\x05\x15\x8e\xe7\xac\x00J\x12*\xf3%J\x10`D\x15\x8a\x05\xcd\xd2L\x96\x00\x8a\x05\xe8\xc7\xf0\xf7\x00\x8a\x05\xe2@t\xa4\x00J\x83\x96\xe1D\x8a\x05&\xed\xf1\xfb\x00\x8a\x05\xa7l\xe1\xd1\x00J\xee\x97z;J\x01\xa4ZiJ\x8d\xbdMTJ\x02\x9a\x8d\t\x8a\x05_n\xd2\xef\x00J\x88\xb0\x0fqJ\xe1v"$J~\xc9\nRJ\xa4\x9c.8\x8a\x05/\xaai\xb4\x00JK;\x992\x8a\x05\xa6\x99`\xcb\x00\x8a\x05nY-\xff\x00\x8a\x05\x82\x81\xeb\xb1\x00\x8a\x05\r\xdb\x88\xbc\x00\x8a\x05\xe4@f\x90\x00JY3\x87LJ\xbd?)J\x8a\x05\x13\x8dT\xbb\x00J\'\xa6\xe4\x1a\x8a\x05\xc9EX\xde\x00\x8a\x05C \x1a\x86\x00\x8a\x05\x8b\x18m\xe6\x00JB=\xa9/\x8a\x05\xd3\xf0\xac\xe4\x00J\xa2d\xa5{\x8a\x05\x11\xda\x8e\xa8\x00J\x05\x8b\xcfnJ\xf6\xe2C\x1f\x8a\x05\xed\xf8\xe8\xe0\x00J\xd2\xdc\xf5}\x8a\x05f\x15\xed\xde\x00\x8a\x05\xde\x86_\xb8\x00J\x89P\x96uJ\xbd\xef\xedPJ/\xc8h J\xab\xec\xb4.\x8a\x05\xee\xe16\x95\x00\x8a\x05gG\x88\xf8\x00\x8a\x05\x96$\x03\xa9\x00J\x08\xe3T-J\xcae\xf2HJ\x8c*\x10&\x8a\x05>p\x7f\xcf\x00J:\xe3ry\x8a\x05y\xa2\xe5\x80\x00J\x03\xfc\x9d\x07J\x05\xbah5J\xf0\xb3\xfaiJ$\x1f\xdf(\x8a\x05\xdeAH\xd9\x00\x8a\x05\x9a\xdck\xe3\x00J\xc2\x01/!J\x90Y\xde+\x8a\x05\xcd\x92"\x86\x00\x8a\x05Z\x01"\xb2\x00\x8a\x05\xef\x07\xa3\xbd\x00J\x82e\xc5cJ(&\xa4\x08J\xbd\x9d\xd1~\x8a\x05\x7f]\x0c\xa5\x00J+\xe4\x98~J\x04\x97\x8b\x15\x8a\x05+\xfe\x1d\x87\x00\x8a\x05~\x88`\xc5\x00\x8a\x05\x03\x92\x85\xfc\x00\x8a\x05S\xe2&\xbf\x00\x8a\x05O\xe6-\x92\x00J\xb6\xe3\n\x14J\x98\x84\'\x00J\xa5\xfb\x9db\x8a\x05\xa0*\xcb\xd0\x00\x8a\x05\x16>?\xc9\x00J`\x1b\x98YJ\x93l\xecHJT3\xfcSJ\xca\x95?\x04J\x80\x8f\x83PJ\xf8\t\xc0A\x8a\x05\x9d\xe2\xee\x9b\x00\x8a\x05\xab6,\x90\x00Jg\x1c`sJ\x94\xb2+bJ\x0f\x8a\x0esJ\x03mj8J\xec\x15\xa4(\x8a\x05v\x89R\x99\x00J(\xaer\x14J*\xc1\xc2|Jl\xdeXW\x8a\x05\xb5\xe1\xfd\x85\x00\x8a\x05\x9c"=\x9b\x00JbNzz\x8a\x05\x17\xe8\n\xda\x00\x8a\x05\xee\\\x8e\xa0\x00J\x15n\xac\x7f\x8a\x05\xd0\xc2`\x94\x00\x8a\x05\xf0\xa3\x19\xbc\x00\x8a\x05\xdd\x1b\xff\xe4\x00J-\n\x15\x08J\x1e\xc16\x7fJB$AtJ0\xaez\x17\x8a\x05\x8fK\x16\xff\x00\x8a\x05+\x92\x95\xe9\x00J3\x0b\xd6 \x8a\x05\xb7\x1b\xe2\xdb\x00J%"\xf2MJ\xac+\xe6<JY4\x9d!Ju\t\xe4\x0fJ9\xf1\xeaIJ\xa3whO\x8a\x05@\xa5\x91\x80\x00\x8a\x05\xfe.\xa5\xe8\x00\x8a\x05e!\xcc\x87\x00\x8a\x05\x11c\xa1\xa9\x00\x8a\x05\xc1\xa1\x0b\x98\x00\x8a\x05\xa2\xfd\xeb\x8b\x00\x8a\x05I\xb4\xa4\xeb\x00J\xc1Q\x06\x1b\x8a\x05\x9a\xf6S\x9d\x00\x8a\x05\xf9\x19i\xe5\x00\x8a\x05\x1d\xb6\x9c\x8d\x00\x8a\x05}\xecE\xd4\x00JT\x94\xb0tJ\xc2\x12\xbec\x8a\x05\xd0\x9c\x13\x8e\x00\x8a\x05\x85\xb4\x06\xaa\x00\x8a\x05\xfb\xbf\x85\xa1\x00J\xc3\xc1E{\x8a\x05\xbfy\xc6\xd2\x00JO\xd8D}JkZ@\x07\x8a\x05\t\x7f\xe6\xa6\x00J\x14\x87\xc1:JD\x02\x03\x04J\xd7\x9e\x1c[J\xbc\x91\xc3\x1aJ\x8a(-S\x8a\x05\x0e\x89\xc2\xd3\x00\x8a\x05X\xb5\xf8\xa1\x00\x8a\x05f\x12\x03\xd9\x00\x8a\x05\xa0y2\xfe\x00JhA\xa4\x00J\x93`\rCJ\xb3\xacx`\x8a\x05I\xf9\xa2\x94\x00\x8a\x05\xec\x90\x8f\xae\x00\x8a\x05\x98\xd8\x89\xde\x00\x8a\x05v\x18j\xd5\x00\x8a\x05\xa9\x03\xa0\x95\x00J\xe3\x1a%hJ\x01R[5Jl5\x11\x7fJ\xc0\xbc\xc4TJ\x9b\xc8x\x19J\xa3\xc4\x86Q\x8a\x05qN,\xf0\x00J[\xff3Z\x8a\x05d|[\xe4\x00\x8a\x05\x14#a\xe6\x00\x8a\x05O\xb2\xe8\xc2\x00\x8a\x05O5\x84\xff\x00\x8a\x05\x95[\xe1\xba\x00J\xe0R1\x11\x8a\x05:V\x1e\xde\x00J%\x9f4\n\x8a\x05\xe9\'\x8e\xfe\x00J\xf5\x816\x17J\xaf\x89s\x1fJX^\x05\x12J\xd2*\xa8Z\x8a\x05\x0fC\xdc\xe7\x00J\xc2\xf4\x89PJ\xb1Z\x8b\x1bJ\x8d7\xb5}\x8a\x05\x85\'\xd8\xf4\x00J\xa7$D\x7f\x8a\x05\xc0\xc0\xf4\xa8\x00\x8a\x05jU\xa8\xfe\x00Jw\xc6w)J\x91\xbb\xf2 \x8a\x05\xec\xdd\xae\xfd\x00\x8a\x05S%T\xce\x00J\x88\xc7i8J\xf4t\x87$J0\x9d\x9cpJ\x1fld(J\xbb\xf3\x87#\x8a\x05\xbb\x05\xd2\xd0\x00\x8a\x05\\6|\x89\x00\x8a\x05\xda\x9b7\xfc\x00J0W\xaa\x03J;1_\x7fJ]\x98a\x07\x8a\x05\x19KQ\xa1\x00J\xce!\x13&J*\x10\x8dRJ\xba\x96\xf1\x19JP\xb5\xecN\x8a\x05\x1d^\xec\x8e\x00JS&\xa6P\x8a\x05"\x95\xbc\xbe\x00JN\x98\xc4A\x8a\x05Y\n\xd3\xf3\x00\x8a\x05\x05\xb3q\xdf\x00J\xd3ub9J<\xec\x95\x12\x8a\x05x\xf0\xcd\xba\x00J\x8a\x85\x9e\x10J6\xf0\x9b4JOn\xa72J\x0f\x9el\\J\x19\xcbQ*\x8a\x05V\xa3m\x8b\x00J\xfcP8\x06\x8a\x05X\xac\x11\x8c\x00\x8a\x05\x970\x0b\xf4\x00\x8a\x05\xa3u\xa4\x9c\x00\x8a\x05\xca6\xeb\xa9\x00J\xac,\x12%J\xc2|x\x07J\x0e\t9FJ^\xe7\x8a\x06J\x02\xbe\x0bD\x8a\x05\\59\xe2\x00J\x05<\xf2\x05J\x16:\x9c\x0e\x8a\x05\xc9\x7f\xc9\xbb\x00\x8a\x05s\xcb\xb4\xf4\x00J\x10{}.J\xd8u\xa9aJ\x0cU\x95|\x8a\x05XU0\xca\x00\x8a\x05\x9d\x0f\x17\xfc\x00J\xd8m\x1b`\x8a\x05Q&M\xb7\x00\x8a\x05NA\x88\xc0\x00J2+\xbd^\x8a\x05o\xcfs\x9b\x00\x8a\x05\x16|\x7f\xbc\x00J28\x05#J*e\x16MJA\t\xc9FJ\xbdz\x95PJ0\xd4T?J\x14B\\k\x8a\x05\xf8\x12Q\xca\x00J\xf8\xd9\x85hJ\xc8O\x89|J\xe2\x0c\xb6\x0c\x8a\x05\xe8\xe6\xf7\xe1\x00\x8a\x05%\x80R\xfd\x00J\xf7\x1a2\x15J\xe1\xc1\x8bw\x8a\x05\x0f\x90\xb6\xe9\x00J\x7f\xef7X\x8a\x05\x982_\xd3\x00J\xbd\xd4[\x05\x8a\x05H{\x1e\xfb\x00J\x82\x89\xfaF\x8a\x05\x11\xc7\x9f\xe0\x00J\xacWT0\x8a\x05r\xdbm\xaf\x00J4G\x97d\x8a\x05>X,\xd3\x00\x8a\x05\xfd\xb7U\x9c\x00\x8a\x05\xa2m\xc6\xa2\x00\x8a\x05\xf4\xa4M\xf6\x00\x8a\x05I\xe0\x84\xc3\x00J\xfd&\x85Q\x8a\x05\xb3G\x11\xee\x00J\xc9^>L\x8a\x05\x1b\xaek\xb3\x00K\xe9t\x94N\x87\x94.'
  #random.setstate(pickle.loads(b))
  random.seed(6)
  graphs = []
  for i in range(10000):
    if (i % 25 == 0):
      print(i)
    #print(i, end=": ")
    g = superminimal_maker(14)
    if (g):
      graphs += [g]

  final = []
  for g in graphs:
    iso = False
    for l in final:
      if (compgraph(g, l)):
        iso = True
    if not iso:
      final += [g]
  print(f"Purged {len(graphs) - len(final)} duplicates. {len(final)} remaining.")
  print("FINAL GRAPHS ARE: \n\n\n")
  for g in final:
    print(g.edge_list(), end=',\n')
      
  #print(pickle.dumps(random.getstate()))


#parses and finds mpgs complements from file of graph6 starts from specific number in case code is interupted
#v is number of vertices
#continuously writes to output file in case code takes too long
def parse_MPG_complement_graph6_partial (inputFile, outputFile, v, start, startIndex):
  input= open(inputFile, "r")
  #append or write?
  output = open(outputFile, "a")
  count=1
  currentGraph6 = input.readline().rstrip('\n')
  #add current mpgs
  mpgs = []
  countDone=0
  while(len(currentGraph6)>0):
    #print(currentGraph6)
    if (countDone<start):
      currentGraph6 = input.readline().rstrip('\n')
      continue
    g = nx.from_graph6_bytes(bytes(currentGraph6, encoding="ascii"))

    adjMatrix =[]
    #print(countDone)
    for i in range (v):
      current =[]
      for j in range (v):
        if g.has_edge(i,j):
          current.append(1)
        else:
          current.append(0)
      adjMatrix.append(current)

    current2 = construct_from_adj(adjMatrix)
    e = check_mpg_complement(current2, True, True) 
    countDone=countDone+1;
    if (countDone%10000==0):
      print("Done: ", countDone)
    if  (e):
      add=True
      if (countDone<=start+10000):
        
        for g in mpgs:
          if g==current2:
            add=False
            break
      
      if(not add):
        currentGraph6 = input.readline().rstrip('\n')
        continue
      mpgs.append(current2)
      
      adj = current2.adjMatrix
      line = str(count)+":"
      for j in range (len(adj)):
        line+=" "+''.join(str(x) for x in adj[j])    
      count=count+1
      output.write(line+'\n')
      output.flush()
    
  
    currentGraph6 = input.readline().rstrip('\n')



  input.close()
  output.close()
  return mpgs



#parses and finds mpgs complements from file of graph6
#v is number of vertices
#continuously writes to output file in case code takes too long
def parse_MPG_complement_graph6_partial (inputFile, outputFile, v):
  input= open(inputFile, "r")
  output = open(outputFile, "w")
  count=1
  currentGraph6 = input.readline().rstrip('\n')
  #print(currentGraph6)
  mpgs = []
  countDone=0
  while(len(currentGraph6)>0):
    #print(currentGraph6)
    g = nx.from_graph6_bytes(bytes(currentGraph6, encoding="ascii"))

    adjMatrix =[]
    #print(countDone)
    for i in range (v):
      current =[]
      for j in range (v):
        if g.has_edge(i,j):
          current.append(1)
        else:
          current.append(0)
      adjMatrix.append(current)

    current2 = construct_from_adj(adjMatrix)
    e = check_mpg_complement(current2, True, True) 
    countDone=countDone+1;
    if (countDone%10000==0):
      print("Done: ", countDone)
    if  (e):
      mpgs.append(current2)
      
      adj = current2.adjMatrix
      line = str(count)+":"
      for j in range (len(adj)):
        line+=" "+''.join(str(x) for x in adj[j])    
      count=count+1
      output.write(line+'\n')
      output.flush()
    
  
    currentGraph6 = input.readline().rstrip('\n')

  input.close()
  output.close()
  return mpgs


#parses mpgs complements from file of graph6 
def parse_MPG_complement_graph6 (file, v):
  graphs = nx.read_graph6(file)
  mpgs = []
  countDone=0
  for g in graphs:
    adjMatrix =[]
    #print(countDone)
    for i in range (v):
      current =[]
      for j in range (v):
        if g.has_edge(i,j):
          current.append(1)
        else:
          current.append(0)
      adjMatrix.append(current)

    current = construct_from_adj(adjMatrix)
    e = check_mpg_complement(current, True, True) #use check_mpg_complement_inner to optimize
    countDone=countDone+1;
    if (countDone%1000==0):
      print("Done: ", countDone, " done - ", math.trunc(math.ceil(1000*countDone/len(graphs)))/10, "%")
    if  (e):
      mpgs.append(current)


  return mpgs

  

#parses mpg complements from file of adjacency matrices
def parse_MPG_complement (file):
  f = open(file, "r")
  row = f.readlines()
  graphs = []
  
  countAdd = 0
  for line in row:
    adjMatrix =[]
    adjRow = line.split()
    countAdd=countAdd+1
    if (countAdd%1000==0):
      print("Add: ", countAdd, " done - ", math.trunc(math.ceil(100*countAdd/len(row))), "%")

    
    for i in range (1,len(adjRow)):
      
      adjMatrix.append(list(map(int, adjRow[i])))
      
    graphs +=[construct_from_adj(adjMatrix)]
  
  countDone = 0
  #graphs[0].print_matrix()
  mpgs2=[]
  for g in graphs:
    e = check_mpg_complement(g, True, True) #use check_mpg_complement_inner to optimize
    countDone=countDone+1;
    if (countDone%1000==0):
      print("Done: ", countDone, " done - ", math.trunc(math.ceil(100*countDone/len(row))), "%")
    if  (e):
      mpgs2.append(g)

  f.close()

  return mpgs2


#satisfies all conditions except for 3-colorable (checks for pseudoMPGs), used to check 5-cycle conjecture
#file in adj matrix
def parse_graph_complement (file):
  f = open(file, "r")
  row = f.readlines()
  mpgs = []
  countAdd = 0
  for line in row:
    adjMatrix =[]
    adjRow = line.split()
    countAdd=countAdd+1
    if (countAdd%1000==0):
      print("Add: ", countAdd, " done - ", math.trunc(math.ceil(100*countAdd/len(row))), "%")

    
    for i in range (1,len(adjRow)):
      
      adjMatrix.append(list(map(int, adjRow[i])))
      
    mpgs +=[construct_from_adj(adjMatrix)]
  
  countDone = 0
  mpgs2=[]
  for g in mpgs:
    if (is_complement_connected(g)):
      if (check_can_add_graph(g)==(-1,-1)):
        mpgs2.append(g)

    countDone=countDone+1;
    if (countDone%1000==0):
      print("Done: ", countDone, " done - ", math.trunc(math.ceil(100*countDone/len(row))), "%")
    
  f.close()
  return mpgs2

#satisfies all conditions except for 3-colorable , used to check 5-cycle conjecture
#checks for pseuoMPGs with file in graph6 format
def parse_graph_complement_graph6 (file, v):
  graphs = nx.read_graph6(file)
  graph = []
  countDone=0
  for g in graphs:
    adjMatrix =[]
    #print(countDone)
    for i in range (v):
      current =[]
      for j in range (v):
        if g.has_edge(i,j):
          current.append(1)
        else:
          current.append(0)
      adjMatrix.append(current)

    
    countDone=countDone+1;
    if (countDone%1000==0):
      print("Done: ", countDone, " done - ", math.trunc(math.ceil(1000*countDone/len(graphs)))/10, "%")
    gr = construct_from_adj(adjMatrix)
    if (is_complement_connected(gr)):
      if (check_can_add_graph(gr)==(-1,-1)):
        graph.append(gr)
  return graph  
 

#find if edge can be added given graph and coloring. returns edge if possible, otherwise (-1, -1)
def check_can_add_graph(g, fast=True, silent=True, permutation=None):
  if permutation == None:
    permutation = range(g.size)
  can_add = (-1, -1)
  for k in range(len(permutation)):
    u = permutation[k]
    for l in range(k+1,len(permutation)):
      v = permutation[l]
      if (g.adjMatrix[u][v] == 0):
        g.adjMatrix[u][v] = 1
        if check_triangle_free(g):
          if not silent:
            print("Added edge between nodes", str(u), "and", str(v))
          can_add = (u, v)
          if (fast): return can_add
        g.adjMatrix[u][v] = 0
  return can_add

#checks for specific type of subgraph, idk why its important
def checkG8():
  
  for i in range (5, 14):
    mpgs = parse_MPG_complement(r"HSMCResearch2024/mpg/Graphs/MPG"+str(i)+"Vertices")
    countFound=0
    countGraphsHaveG8=0
    for g in mpgs:
      found=False
      for (u, v) in g.edge_list():
        
        for w in range(g.size):
          if (g.adjMatrix[v][w] != 1 or w in  [u,v]):
            continue
          for x in range(g.size):
            
            if (g.adjMatrix[u][x]!=1 or g.adjMatrix[w][x] != 1 or x in [u, v,w]):
              continue
            for y in range(g.size):
              if (y in [u, v, w, x]):
                continue
              if (g.adjMatrix[x][y] == 1 and g.adjMatrix[v][y]==1):
                found=True
                countFound=countFound+1
      if (found):
        countGraphsHaveG8=countGraphsHaveG8+1


    print("There are ", str(countGraphsHaveG8), " MPGs with ", str(i), "vertices that have G8 subgraphs with ", str(countFound), "total possibly isomorphic G8s." )

#checks for 4 cycles in mpg complements, used to disprove 4 cycle conjecture        
def check_4_cycle(file):
  for i in range (5,14):
    mpgs = parse_MPG_complement(file+str(i)+"Vertices")
    countFound = 0
    countFailed=0
    totalFound=0
    for g in mpgs:
      found = False
      for (u, v) in g.edge_list():
        
        for w in range(g.size):
          if (g.adjMatrix[v][w] != 1 or w in [u,v]):
            continue
          for x in range(g.size):
            if (x in [u,v,w]):
              continue
            if (g.adjMatrix[w][x] == 1 and g.adjMatrix[x][u]==1):
              found = True
              totalFound=totalFound+1
      if (found):
        countFound=countFound+1
      else:
        countFailed=countFailed+1
        if (i==10 and check_mpg_complement(g, True, True)):
          print(g.edge_list())

    print(str(i), " Vertices: ", countFound, " found with ", totalFound, "(possibly isomorphic) cycles and ", countFailed, " graphs without a 4-cycle.")


#check 6 cycle conjecture - at least one vertex connects 2 opposite edges
def check_6_cycle_conjecture(g,c, u,v,w,x,y,z):
  #print("here")
  ux = False
  vy = False
  wz = False
  for i in range(g.size):
    if (i in [u,v,w,x,y,z]):
      continue

    if (g.adjMatrix[i][u]==1 and g.adjMatrix[i][x]==1):
      ux = True

    if (g.adjMatrix[i][v]==1 and g.adjMatrix[i][y]==1):
      vy = True

    if (g.adjMatrix[i][w]==1 and g.adjMatrix[i][z]==1):
      wz = True

    if (ux or vy or wz):
      return True
    
  return False


#check 6 cycle with coloring
def check_6_cycle(g,c ):
  
  count_6_cycle_conjecture_works=0
  for (u, v) in g.edge_list():
      cycle = False
      for w in range(g.size):
        if (g.adjMatrix[v][w] != 1 or w in [u,v]):
          continue
        for x in range(g.size):
          if (g.adjMatrix[w][x] != 1 or x in [u, v,w]):
            continue
          for y in range(g.size):
            if (g.adjMatrix[x][y] != 1 or y in [u, v, w, x]):
              continue
            for z in range (g.size):
              if (z in [u,v,w,x,y]):
                continue
              if (g.adjMatrix[y][z]==1 and g.adjMatrix[z][u]==1):
                cycle = True
                if (c[u]==c[x] and c[v]==c[y] and c[w]==c[z]):
                  # print("here2")
                  if (not check_6_cycle_conjecture(g,c,u,v,w,x,y,z)):
                    if (check_mpg_complement(g,True,True)):
                      print(g.edge_list())
                      print(u,v,w,x,y,z)
                      print(c)
                      print ("Conjecture is false")
                      print(" ")
                    
                    

                  # else:
                  #   print("Conjecture true")
                  #   count_6_cycle_conjecture_works=count_6_cycle_conjecture_works+1

                  # print(g.edge_list())
                  # print(u,v,w,x,y,z)
                  # print(c)
                  # print(" ")
  #return count_6_cycle_conjecture_works


#used to check if 6 cycle alternating coloring exists
def check_mpg_complement_inner_6_cycle(g, fast=False, silent=False):
  if not silent:
    print("Valid coloring list:")
  colorings = sat_colorability_solver.allSolutions(g.size, g.edge_list())
  if colorings == []:
    return None # this means no coloring found

  edge_add = []
  for coloring in colorings:
    if (not silent): print(coloring)
    while (edge := check_can_add(g, coloring, fast=True, silent=silent)) != (-1, -1):
      edge_add += [edge]
      if fast: 
        return edge_add
      
    if is_complement_connected(g):
      if check_triangle_free(g):
        
        check_6_cycle(g,coloring)
 
  return edge_add

#used to check if 6 cycle alternating coloring exists
def check_mpg_complement_6_cycle(g, fast=False, silent=False):
  t1 = time.time()
  if not is_complement_connected(g):
    if not silent:
      print("MPG is not connected.")
    return False
  if check_triangle_free(g):
    minimal = check_mpg_complement_inner_6_cycle(g, fast=fast, silent=silent)
    

    t2 = time.time()
    if not silent:
      print("Time taken:", str(t2 - t1), "seconds")
    if (minimal == None):
      if not silent:
        print("Graph is not colorable")
      return False
    if (minimal == []):
      if not silent:
        print("Graph is minimal")
      return True
    else:
      if not silent:
        print("Not minimal")
      return False
  if not silent:
    print("Not triangle free")
  t2 = time.time()
  if not silent:
    print("Time taken:", str(t2 - t1), "seconds")
  return False

#used to output all mpgs in a list to output file of choice
#outputed in adj matrix using input format
def output_MPG(mpg, file):
  text=[]
  count=1
  for g1 in mpg:
    adj = g1.adjMatrix
    line = str(count)+":"
    for j in range (len(adj)):
      line+=" "+''.join(str(x) for x in adj[j])
    text.append(line+'\n')
    count=count+1
    
  file = open(file, "w")
  file.writelines(text)
  file.close()



def main():
  #print(os.path.abspath("mpg/fiveV"))
  #print(os.listdir())
  #print("test")
  # amnt = {}
  # count= 0
  # for m in m1:
  #   m = construct_from_edge_list(m).adjMatrix
  #   #printm(m)
  #   #print()
  #   amnt[len(m)] = amnt.get(len(m), 0)+1
  #   count += 1
  # print(dict(sorted(amnt.items())))
  # print(count)
  #for m in m1:
  #  if len(m) == 10:
  #    print(m)
  #    check_superminimal_complement(construct_from_adj(m))

  #for m in m1:
  #  print(construct_from_adj(m).edge_list())

  #for m in m2:
  #  iso = False
  #  for g in m1:
  #    if (compgraph(construct_from_adj(m), construct_from_adj(g))):
  #      iso = True
  #      break
  #  if iso == False:
  #    print(m, ",")
  #looped_superminimal_finder()
  #print(m1[0])
  
  #printm(g.adjMatrix)
  

  
  #up to number of vertices
  #print (parseMPG("fiveV")[0].adjMatrix)
  #vertices = 10
  #file = str(vertices)+"V"
  #mpg = parse_MPG_complement(file)
  # #graph = parse_graph_complement(file)
  #print("MPG: ", len(mpg))
  #print("Graph: ", len(graph))

  # file_graph6 = str(vertices)+"Vertices"
  # mpgs = parse_MPG_complement_graph6(r"HSMCResearch2024/mpg/nauty2_8_8/TF_C_D2/"+ file_graph6,vertices)
  # print ("MPG from Nauty: ", len(mpgs))

  # for i in range(5,14):
  #   mpgs = parse_MPG_complement(r"HSMCResearch2024/mpg/Graphs/MPG"+str(i)+"Vertices")
  #   graphs = parse_graph_complement(r"HSMCResearch2024/mpg/Graphs/Graphs"+str(i)+"Vertices")
  #   for g in mpgs:
  #     check_mpg_complement_6_cycle(g,False,True)
    # uniqueMPG = 0
    # uniqueGraph =0
    # shared =0
    # shared2=0

    # for g in mpgs:
    #   same = False
    #   for h in graphs:
    #     if (compgraph(g,h)):
    #       same=True;
    #       break
    #   if (not same):
    #     uniqueMPG = uniqueMPG+1
    #   else:
    #     shared=shared+1
    
    # for g in graphs:
    #   same = False
    #   for h in mpgs:
    #     if (compgraph(g,h)):
    #       same=True;
    #       break
    #   if (not same):
    #     uniqueGraph = uniqueGraph+1
    #   else:
    #     shared2=shared2+1
    # print(i,"Vertices: , Total MPGs: ",len(mpgs),", Total Graphs: ",len(graphs))
    # print("Unique MPGs: ", uniqueMPG, ", Unique Graphs: ", uniqueGraph,", Shared: ",shared ,"=",shared2)
  
 
  # fiveVertices = nx.read_graph6(r"HSMCResearch2024/mpg/nauty2_8_8/all5Vertices")
  # graph = []
  
  # for g in fiveVertices:
  #   adjMatrix =[]
  #   #print(countDone)
  #   for i in range (5):
  #     current =[]
  #     for j in range (5):
  #       if g.has_edge(i,j):
  #         current.append(1)
  #       else:
  #         current.append(0)
  #     adjMatrix.append(current)
  #   gr = construct_from_adj(adjMatrix)
  #   print(gr.edge_list())
    


  #   for g1 in graphs:
  #     adj = g1.adjMatrix
  #     line = str(count)+":"
  #     for j in range (i):
  #       line+=" "+''.join(str(x) for x in adj[j])
  #     text.append(line+'\n')
  #     count=count+1
    
  #   file = open(r"Graphs"+str(i)+"Vertices", "w")
  #   file.writelines(text)
  #   file.close()



  mpg = parse_MPG_complement_graph6_partial(r"/Users/yy2442/HSMCResearch2024/Data/SearchSpace/Graph6/14Vertices",r"/Users/yy2442/HSMCResearch2024/Data/MPG14V", 14)
  #print(mpg[0].adjMatrix)
  #print(len(mpg))
  #output_MPG(mpg, r"/Users/yy2442/HSMCResearch2024/Data/MPG14Vertices")
  
  # vertices = 13
  # file = r"HSMCResearch2024/mpg/Graphs/MPG" +str(vertices)+"Vertices"

  # allCycle = True
  # mpg = parse_MPG_complement(file)
  # for g in mpg:
  #   if find5cycle_strong(g):
  #     print("MPGs not all 5 cycle")
  #     allCycle =False
  #     break
    
  #   #print (g.edge_list())
  # if allCycle:
  #   print ("MPGs all 5 cycle")
  #   print(len(mpg))

  # graph = parse_graph_complement_graph6(file_graph6, vertices)
  # allCycleGraph = True
  # for g in graph:
  #   #print(g.edge_list())
  #   if find5cycle_strong(g):
  #     print("Graphs not all 5 cycle")
  #     allCycleGraph =False
  #     break
    
  #   #print (g.edge_list())
  # if allCycleGraph:
  #   print ("Graphs all 5 cycle")
  #   print(len(graph))

  #check_4_cycle(r"HSMCResearch2024/mpg/Graphs/MPG")
  #check_4_cycle(r"HSMCResearch2024/mpg/Graphs/Graphs")

  

  
  

if __name__ == '__main__':
  main()

