import numpy as np
import sys


# program to infer MR from
# i is the array containing the input arguments; o is the array containing the returning results
def program(i, func_index):
    """
    :rtype: array
    :type func_index: int
    :type i: array
    :
    """
    if func_index == 1:
        x1 = i[0]
        x2 = i[1]
        o = np.array([np.hypot(x1, x2)])

    return o


# the number of elements of the input for the programs
def getNEI(func_index):
    """
    :rtype: int
    :type func_index: int
    """
    if func_index in [1]:
        no_of_elements_input = 2

    return no_of_elements_input


# the number of elements of the output for the programs
def getNEO(func_index):
    """
    :type func_index: int
    :rtype: int
    """
    if func_index in [1]:
        no_of_elements_output = 1

    return no_of_elements_output


# domain for each element of the input
def get_input_range(func_index):
    """
    :type func_index: int
    :rtype: int
    """
    if func_index in [1]:
        input_range = np.array([[0, 20], [0, 20]])

    return input_range


# datatype for each element of the input:
# int8	Byte (-128 to 127)
# int16	Integer (-32768 to 32767)
# int32	Integer (-2147483648 to 2147483647)
# int64	Integer (-9223372036854775808 to 9223372036854775807)
# float16	Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
# float32	Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
# float64	Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
def get_input_datatype(func_index):
    """
    :type func_index: int
    :rtype: list
    """
    if func_index in [1]:
        datatype = ["float16", "float16"]

    return datatype


# which programs to infer MRs from
func_indices = [1]

# which type of MRs to infer: NOI_MIR_MOR_DIR_DOR.
# NOI: number of inputs
# MIR, MOR: mode of input and output relations. 1-equal, 2-greaterthan, 3-lessthan
# DIR, DOR: degrees of input and output relations. 1-linear, 2-quadratic, etc.
parameters_collection = ["2_1_1_1_1", "2_1_1_1_2", "2_1_1_1_3", "3_1_1_1_1", "3_1_1_1_2", "2_1_2_1_1", "2_1_3_1_1",
                         "2_2_1_1_1", "2_3_1_1_1", "2_2_2_1_1", "2_2_3_1_1", "2_3_2_1_1", "2_3_3_1_1"]

# path to store results
output_path = "./output/test"

# search parameters
pso_runs = 2
pso_iterations = 2

# set search range for coeff_range, const_range
coeff_range = np.array([-2, 2])
const_range = np.array([-10, 10])
