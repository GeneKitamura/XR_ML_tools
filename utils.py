import numpy as np

def array_print(*args):
    for i in args:
        print(i.shape)

def array_min_max(*args):
    for i in args:
        print(np.min(i), np.max(i))