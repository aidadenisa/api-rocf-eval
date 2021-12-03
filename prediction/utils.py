import ast
import numpy as np

def to_tuple(t):
    return ast.literal_eval(t)

def to_float(a):
    return np.array(a[1:-1].split(',')).astype(float)