#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from itertools import product
from pulp import *
from pyomo.environ import *
from pyomo.opt import SolverFactory


# In[2]:


np.random.seed(1)
ns, nf, ng, nq = 4, 4, 3, 2


# In[3]:


pr_fs = list(product(range(nf), range(ns)))
pr_gs = list(product(range(ng), range(ns)))
pr_ffq = list(product(range(nf), range(nf), range(nq)))
pr_fgq = list(product(range(nf), range(ng), range(nq)))
pr_ffss = list(product(range(nf), range(nf), range(ns), range(ns)))
pr_fgss = list(product(range(nf), range(ng), range(ns), range(ns)))


# In[4]:


sCPU = np.random.randint(32, 33, ns)
fCPU = np.random.randint(0, 1, ns)
sRAM = np.random.randint(96, 128, ns)
fRAM = np.random.randint(0, 4, ns)


# In[5]:


sBW = np.random.randint(100, 200, (ns, ns))
sDelay = np.array([[0, 10, 10, 10], [10, 0, 10, 10], [10, 10, 0, 10], [10, 10, 10, 0]])


# In[6]:


cPrice = np.random.randint(8, 12, ns)
rPrice = np.random.randint(1, 2, ns)
bPrice = np.random.randint(2, 3)


# In[7]:


cAF = np.random.randint(4, 8, nf)
rAF = np.random.randint(2, 4, nf)


# In[8]:


bChain = np.random.randint(1, 20, nq)
dChain = np.random.randint(10, 20, nq)


# In[9]:


sCPU, fCPU, sDelay, cPrice, rPrice, cAF, rAF


# In[10]:


x = {(m, i): 0 for m, i in pr_gs}
x[(0, 0)] = 1 
x[(1, 1)] = 1 
x[(2, 2)] = 1 


# In[11]:


A_f = {(l, m, k): 0 for l, m, k in pr_ffq}
A_f[(0, 1, 0)] = 1
A_f[(1, 0, 0)] = 1
A_f[(1, 2, 0)] = 1
A_f[(2, 1, 0)] = 1
A_f[(0, 3, 1)] = 1
A_f[(3, 0, 1)] = 1


# In[12]:


A_g = {(m, n, k): 0 for m, n, k in pr_fgq}
A_g[(0, 0, 0)] = 1
A_g[(2, 1, 0)] = 1
A_g[(0, 0, 1)] = 1
A_g[(3, 2, 1)] = 1


# In[13]:


M = ConcreteModel("AF allocation")


# In[14]:


M.I = Set(initialize=[(m, i) for m, i in pr_fs])


# In[15]:


M.x = Var(M.I, within=Binary)


# In[16]:


M.value = Objective(expr=sum(M.x[m, i] * cPrice[i] * cAF[m] + M.x[m, i] * rPrice[i] * rAF[m] for m, i in pr_fs), sense=minimize)


# In[17]:


M.const = ConstraintList()


# In[18]:


for m in range(nf):
    M.const.add(sum(M.x[m, i] for i in range(ns)) == 1)


# In[19]:


for i in range(ns):
    M.const.add(sum(M.x[m, i] * cAF[m] for m in range(nf)) <= sCPU[i] - fCPU[i])


# In[20]:


for i in range(ns):
    M.const.add(sum(M.x[m, i] * rAF[m] for m in range(nf)) <= sRAM[i] - fRAM[i])


# In[21]:


for k in range(nq):
    M.const.add(sum(M.x[l, i] * M.x[m, j] * A_f[l, m, k] * sDelay[i, j] for l, m, i, j in pr_ffss if l < m)                 + sum(M.x[m, i] * x[n, j] * A_g[m, n, k] * sDelay[i, j] for m, n, i, j in pr_fgss) <= dChain[k])


# In[22]:


opt = SolverFactory("cplex")
result = opt.solve(M, tee=True)
M.display()


# model = LpProblem()

# y = {(m, i): LpVariable('y%d_%d'%(m, i), cat=LpBinary) for m, i in pr_fs}

# z = {(l, m, i, j): LpVariable('z%d_%d_%d_%d'%(l, m, i, j), cat=LpBinary) for l, m, i, j in pr_ffss}

# model += lpSum(y[(m, i)] * cPrice[i] * cAF[m] + y[m, i] * rPrice[i] * rAF[m] for m, i in pr_fs)

# for m in range(nf):
#     model += lpSum(y[m, i] for i in range(ns)) == 1

# for i in range(ns):
#     model += lpSum(y[m, i] * cAF[m] for m in range(nf)) <= sCPU[i] - fCPU[i]

# for i in range(ns):
#     model += lpSum(y[m, i] * rAF[m] for m in range(nf)) <= sRAM[i] - fRAM[i]

#  for l, m, i, j in pr_ffss:
#     model += z[l, m, i, j] <= y[l, i]
#     model += z[l, m, i, j] <= y[m, j]
# 
# for l, i in pr_fs:
#     model += lpSum(z[l, m, i, j] for m, j in pr_fs) == y[l, i]
# for m, j in pr_fs:
#     model += lpSum(z[l, m, i, j] for l, i in pr_fs) == y[m, j]

# for k in range(nq):
#     model += lpSum(z[l, m, i, j] * A_f[l, m, k] * sDelay[i, j] for l, m, i, j in pr_ffss if l < m) \
#     + lpSum(y[m, i] * x[n, j] * A_g[m, n, k] * sDelay[i, j] for m, n, i, j in pr_fgss) <= dChain[k]

# model.solve()
