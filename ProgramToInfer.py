import numpy as np
import sys

def setClasspath(func_index):
    if func_index > 1000:
        import jnius_config
        jnius_config.set_classpath('.', f'/home/c3288930/Branch_Defects4JOnHPC/mr-infer/AutoMR/d4j_Math/fixed/{func_index-1000}/target/classes')
    else:
        pass

def setM(func_index):
    global m
    from jnius import autoclass
    if func_index == 1003:
        m = autoclass("org.apache.commons.math3.util.MathArrays")
    elif func_index == 1015:
        m = autoclass("org.apache.commons.math3.util.FastMath")
    elif func_index == 1016:
        m = autoclass("org.apache.commons.math3.util.FastMath")
    elif func_index == 1059:
        m = autoclass("org.apache.commons.math.util.FastMath")
    elif func_index == 1063:
        m = autoclass("org.apache.commons.math.util.MathUtils")
    else:
        pass

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
        o = np.array([int(np.equal(i1, i2))])
    elif func_index == 1003:
        i1 = [i[0], i[1]]
        i2 = [i[2], i[3]]
        o = np.array([m.linearCombination(i1, i2)])
    elif func_index == 1015:
        i1 = i[0]
        i2 = i[1]
        o = np.array([m.pow(i1, i2)])
    elif func_index == 1016:
        i1 = i[0]
        o = np.array([m.sinh(i1)])
    elif func_index == 1059:
        i1 = i[0]
        i2 = i[1]
        o = np.array([m.max(i1, i2)])
    elif func_index == 1063:
        i1 = i[0]
        i2 = i[1]
        o = np.array([int(m.equals(i1, i2))])

    return o

# the number of elements of one input for the program
def getNEI(func_index):
    if func_index in [7, 14, 18, 19, 1015, 1059, 1063, 26, 27]:
        no_of_elements_input = 2
    elif func_index in [1003, 28]:
        no_of_elements_input = 4
    else:
        no_of_elements_input = 1

    return no_of_elements_input

# the number of elements of one output for the program
def getNEO(func_index):
    no_of_elements_output = 1

    return no_of_elements_output


def get_name_of_program(func_index):
    if func_index == 1:
        name = "np_abs"
    elif func_index == 2:
        name = "np_acos"
    elif func_index == 3:
        name = "np_acosh"
    elif func_index == 4:
        name = "np_asin"
    elif func_index == 5:
        name = "np_asinh"
    elif func_index == 6:
        name = "np_atan"
    elif func_index == 7:
        name = "np_atan2"
    elif func_index == 8:
        name = "np_atanh"
    elif func_index == 9:
        name = "np_ceil"
    elif func_index == 10:
        name = "np_cos"
    elif func_index == 11:
        name = "np_cosh"
    elif func_index == 12:
        name = "np_exp"
    elif func_index == 13:
        name = "np_floor"
    elif func_index == 14:
        name = "np_hypot"
    elif func_index == 15:
        name = "np_log"
    elif func_index == 16:
        name = "np_log1p"
    elif func_index == 17:
        name = "np_log10"
    elif func_index == 18:
        name = "np_max"
    elif func_index == 19:
        name = "np_min"
    elif func_index == 20:
        name = "np_round"
    elif func_index == 21:
        name = "np_sin"
    elif func_index == 22:
        name = "np_sinh"
    elif func_index == 23:
        name = "np_sqrt"
    elif func_index == 24:
        name = "np_tan"
    elif func_index == 25:
        name = "np_tanh"
    return name



# which programs to infer MRs from
func_indices = [i for i in range(1, 26)]

# which type of MRs to infer: NOI_MIR_MOR_DIR_DOR.
# NOI: number of inputs
# MIR, MOR: mode of input and output relations. 1-equal, 2-greaterthan, 3-lessthan
# DIR, DOR: degrees of input and output relations. 1-linear, 2-quadratic, etc.
parameters_collection = ["2_1_1_1_1", "2_1_1_1_2", "2_1_1_1_3", "3_1_1_1_1", "3_1_1_1_2", "2_1_2_1_1", "2_1_3_1_1", "2_2_1_1_1", "2_3_1_1_1", "2_2_2_1_1", "2_2_3_1_1", "2_3_2_1_1", "2_3_3_1_1"]

# path to store results
output_path = "./output/extraNp"

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