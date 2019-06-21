import numpy as np
import pandas as pd
from itertools import product
from pulp import *

np.random.seed(1)
ns, nf, nq = 4, 4, 2
pr = list(product(range(ns),range(ns)))

sCPU = np.random.randint(10, 20, ns)
fCPU = np.random.randint(0, 10, ns)
sRAM = np.random.randint(64, 128, ns)
fRAM = np.random.randint(0, 64, ns)

sBW = np.random.randint(10, 100, (ns, ns))
sDelay = np.random.randint(10, 100, (ns, ns))

cPrice = np.random.randint(5, 15, ns)
rPrice = np.random.randint(1, 3, ns)
bPrice = np.random.randint(2, 2, 1)

cAF = np.random.randint(1, 4, nf)
rAF = np.random.randint(2, 16, nf)

bChain = np.random.randint(1, 20, nq)
dChain = np.random.randint(100, 300, nq)