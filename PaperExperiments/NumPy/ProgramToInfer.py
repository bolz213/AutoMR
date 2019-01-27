import numpy as np
import sys

# program to infer MR from， i is a list containing the arguments: for example, program([2, 3], 18)
def program(i, func_index):
    if func_index == 1:
        o = np.abs(i)
    elif func_index == 2:
        o = np.arccos(i)
    elif func_index == 3:
        o = np.arccosh(i)
    elif func_index == 4:
        o = np.arcsin(i)
    elif func_index == 5:
        o = np.arcsinh(i)
    elif func_index == 6:
        o = np.arctan(i)
    elif func_index == 7:
        o = np.arctan2(i[0], i[1])
    elif func_index == 8:
        o = np.arctanh(i)
    elif func_index == 9:
        o = np.ceil(i)
    elif func_index == 10:
        o = np.cos(i)
    elif func_index == 11:
        o = np.cosh(i)
    elif func_index == 12:
        o = np.exp(i)
    elif func_index == 13:
        o = np.floor(i)
    elif func_index == 14:
        o = np.hypot(i[0], i[1])
    elif func_index == 15:
        o = np.log(i)
    elif func_index == 16:
        o = np.log(i + 1)
    elif func_index == 17:
        o = np.log10(i)
    elif func_index == 18:
        o = np.amax(i)
    elif func_index == 19:
        o = np.amin(i)
    elif func_index == 20:
        o = np.round(i)
    elif func_index == 21:
        o = np.sin(i)
    elif func_index == 22:
        o = np.sinh(i)
    elif func_index == 23:
        o = np.sqrt(i)
    elif func_index == 24:
        o = np.tan(i)
    elif func_index == 25:
        o = np.tanh(i)
    elif func_index == 26:
        i1 = [i[0]]
        i2 = [i[1]]
        o = np.array([np.dot(i1, i2)])
    elif func_index == 27:
        i1 = [i[0]]
        i2 = [i[1]]
        o = np.array([np.power(i1, i2)])
    elif func_index == 28:
        i1 = [i[0], i[1]]
        i2 = [i[2], i[3]]
        o = np.array([int(np.array_equal(i1, i2))])
    elif func_index == 29:
        i1 = [i[0]]
        i2 = [i[1]]
        o = np.array([np.sort(i1, i2)])

    return o

# the number of elements of one input for the program
def getNEI(func_index):
    if func_index in [7, 14, 18, 19, 26, 27, 29]:
        no_of_elements_input = 2
    elif func_index in [28]:
        no_of_elements_input = 4
    else:
        no_of_elements_input = 1

    return no_of_elements_input

# the number of elements of one output for the program
def getNEO(func_index):
    if func_index == 29:
        no_of_elements_output = getNEI(func_index)
    else:
        no_of_elements_output = 1
    return no_of_elements_output

# which programs to infer MRs from
func_indices = [i for i in range(1, 30)]

# which type of MRs to infer: NOI_MIR_MOR_DIR_DOR.
# NOI: number of inputs
# MIR, MOR: mode of input and output relations. 1-equal, 2-greaterthan, 3-lessthan
# DIR, DOR: degrees of input and output relations. 1-linear, 2-quadratic, etc.
parameters_collection = ["2_1_1_1_1", "2_1_1_1_2", "2_1_1_1_3", "3_1_1_1_1", "3_1_1_1_2", "2_1_2_1_1", "2_1_3_1_1", "2_2_1_1_1", "2_3_1_1_1", "2_2_2_1_1", "2_2_3_1_1", "2_3_2_1_1", "2_3_3_1_1"]

# path to store results
output_path = "./output/numpy"

# search parameters
pso_runs = 500
pso_iterations = 350

# range to generate test cases
def get_input_range(func_index):
    if func_index:
        r = [0, 20]
        input_range = np.tile(r, (getNEI(func_index), 1))

    return input_range

# set search range for coeff_range, const_range
coeff_range = np.array([-5, 5])
const_range = np.array([-10, 10])
