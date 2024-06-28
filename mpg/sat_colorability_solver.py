#finds 3-colorings

from pysat.formula import CNF
from pysat.solvers import Solver
def solve(vertex_count, edge_list): 
  arr = []
  for i in range(vertex_count):
    r = 2*i+1
    b = 2*i+2
    #g = r off and b off
    arr += [[-r, -b]] #cannot be both red and blue

  for edge in edge_list:
    [v1, v2] = edge
    r1 = 2*v1+1
    b1 = 2*v1+2
    r2 = 2*v2+1
    b2 = 2*v2+2
    arr += [[-r1, -r2], [-b1, -b2], [r1, b1, r2, b2]]

  adjacent_vertex = -1
  for edge in edge_list: 
    [v1, v2] = edge
    if (v1 == 0):
      adjacent_vertex = v2
      break
    elif (v2 == 0):
      adjacent_vertex = v1
      break

  bootstrap = [[1],[-2]]
  if (adjacent_vertex != -1):
    bootstrap += [[-2 * adjacent_vertex - 1], 
                  [2 * adjacent_vertex + 2]]

  s = Solver(bootstrap_with=bootstrap)
  for condition in arr:
    s.add_clause(condition)
  return s

def allSolutions(v, e):
  s = solve(v, e)
  arr = []
  while (s.solve()):
    m = s.get_model()
    assert m
    arr += [m]
    s.add_clause([-i for i in m])

  out = []
  for entry in arr:
    new = []
    for i in range(len(entry)//2):
      new += [0 if entry[2*i] > 0 else (1 if entry[2*i + 1] > 0 else 2)]
    out += [new]
  return out
