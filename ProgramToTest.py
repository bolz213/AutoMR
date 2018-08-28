import numpy as np
import os


func_indices = [1]

# path of the dir where the MRs are put
MRs_path = "./output/temp"


# number of programs to test
def get_number_of_mutants(func_index):
    if func_index == 1:
        return 2


# program to test
def get_mutant(func_index):
    def func1(i, index_mutant):
        if index_mutant == 0:
            x1 = i[0]
            x2 = i[1]
            o = np.array([np.hypot(x1, x2)])
        elif index_mutant == 1:
            o = np.array([np.max(i)])

        return o

    if func_index == 1:
        return func1



