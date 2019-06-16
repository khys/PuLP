import numpy as np
import pandas as pd
from itertools import product
from pulp import *

np.random.seed(1)
nw, nf = 3, 4
pr = list(product(range(nw), range(nf)))
供給 = np.random.randint(30, 50, nw)
需要 = np.random.randint(20, 40, nf)
輸送費 = np.random.randint(10, 20, (nw, nf))