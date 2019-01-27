import numpy as np
import jnius_config
jnius_config.set_classpath('.', './mutants/bin')

from jnius import autoclass


# program to infer MR from， i is a list containing the arguments: for example, program([2, 3], 18)
def program(i, func_index):
    if func_index == 26:
        Mu = autoclass("sin_mutants.sin_original")
        mu = Mu()
        i1 = np.float(i[0])
        o = np.array([mu.Sin(i1)])
    elif func_index == 27:
        Mu = autoclass("pku.abs")
        mu = Mu()
        i1 = i[0]
        o = np.array([mu.abs(i1)])
    elif func_index == 28:
        Mu = autoclass("pku.asinh")
        mu = Mu()
        i1 = i[0]
        o = np.array([mu.asinh(i1)])
    elif func_index == 29:
        Mu = autoclass("pku.atan")
        mu = Mu()
        i1 = i[0]
        o = np.array([mu.atan(i1)])
    elif func_index == 30:
        Mu = autoclass("cos_mutants.cos_original")
        mu = Mu()
        i1 = i[0]
        o = np.array([mu.Cos(i1)])
    elif func_index == 31:
        func_name = "log1p"
        Mu = autoclass(f"pku.{func_name}")
        mu = Mu()
        i1 = i[0]
        o = np.array([mu.log1p(i1)])
    elif func_index == 32:
        func_name = "log10"
        Mu = autoclass(f"pku.{func_name}")
        mu = Mu()
        i1 = i[0]
        o = np.array([mu.log10(i1)])
    elif func_index == 33:
        mu_Tan = autoclass(f"tan_mutants.tan_original")
        mu_tan = mu_Tan()
        i1 = np.float(i[0])
        o = np.array([mu_tan.tan(i1)])
    return o

# the number of elements of one input for the program
def getNEI(func_index):
    if func_index in [7, 14, 18, 19]:
        no_of_elements_input = 2
    else:
        no_of_elements_input = 1
    return no_of_elements_input

# the number of elements of one output for the program
def getNEO(func_index):
    no_of_elements_output = 1
    return no_of_elements_output



# which programs to infer MRs from
func_indices = [26, 27, 28, 29, 30, 31, 32, 33]

# which type of MRs to infer: NOI_MIR_MOR_DIR_DOR.
# NOI: number of inputs
# MIR, MOR: mode of input and output relations. 1-equal, 2-greaterthan, 3-lessthan
# DIR, DOR: degrees of input and output relations. 1-linear, 2-quadratic, etc.
parameters_collection = ["2_1_1_1_1", "2_1_1_1_2", "2_1_1_1_3", "3_1_1_1_1", "3_1_1_1_2", "2_1_2_1_1", "2_1_3_1_1", "2_2_1_1_1", "2_3_1_1_1", "2_2_2_1_1", "2_2_3_1_1", "2_3_2_1_1", "2_3_3_1_1"]

# path to store results
output_path = "./output/apache"

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
