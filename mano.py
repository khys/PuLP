import numpy as np
#import pandas as pd
from itertools import product
from pulp import *

np.random.seed(1)
ns, nf, ng, nq = 4, 4, 3, 2
pr_fs = list(product(range(nf),range(ns)))
pr_gs = list(product(range(ng),range(ns)))
pr_ffq = list(product(range(nf),range(nf), range(nq)))
pr_fgq = list(product(range(nf),range(ng), range(nq)))

sCPU = np.random.randint(10, 20, ns)
fCPU = np.random.randint(0, 10, ns)
sRAM = np.random.randint(64, 128, ns)
fRAM = np.random.randint(0, 64, ns)

# 同一 Spot 間は 0 にしないとダメ
sBW = np.random.randint(10, 100, (ns, ns))
sDelay = np.random.randint(10, 100, (ns, ns))

cPrice = np.random.randint(5, 15, ns)
rPrice = np.random.randint(1, 3, ns)
# bPrice = np.random.randint(2, 2, 1)

cAF = np.random.randint(1, 4, nf)
rAF = np.random.randint(2, 16, nf)

bChain = np.random.randint(1, 20, nq)
dChain = np.random.randint(100, 300, nq)

x = {(m, i): 0 for m, i in pr_gs}
x[(0, 0)] = 1 
x[(1, 1)] = 1 
x[(2, 2)] = 1 
A_f = {(l, m, k): 0 for l, m, k in pr_ffq}
A_f[(0, 1, 0)] = 1
A_f[(1, 0, 0)] = 1
A_f[(1, 2, 0)] = 1
A_f[(2, 1, 0)] = 1
A_f[(0, 3, 1)] = 1
A_f[(3, 0, 1)] = 1
A_g = {(m, n, k): 0 for m, n, k in pr_fgq}
A_g[(0, 0, 0)] = 1
A_g[(2, 1, 0)] = 1
A_g[(0, 0, 1)] = 1
A_g[(3, 2, 1)] = 1

model = LpProblem()
y = {(m, i): LpVariable('y%d_%d'%(m, i), cat=LpBinary) for m, i in pr_fs}
model += lpSum(y[(m, i)] * cPrice[i] * cAF[m] + y[m, i] * rPrice[i] * rAF[m] for m, i in pr_fs)
for m in range(nf):
    model += lpSum(y[m, i] for i in range(ns)) == 1
model.solve()