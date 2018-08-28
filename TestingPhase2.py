import os
import sys
import numpy as np
import pandas as pd
import pickle

import Phase1_PSOSearch
import Phase2_Filter
import ProgramToTest
import ProgramToInfer
import Phase3_RemoveRedundancy


def main():
    for func_index in func_indices:
        stats = pd.DataFrame()
        program = ProgramToTest.get_program(func_index)
        number_of_mutants = ProgramToTest.get_number_of_mutants(func_index)
        no_of_elements_output = ProgramToInfer.getNEO(func_index)
        no_of_elements_input = ProgramToInfer.getNEI(func_index)
        inputcases_range = np.tile(ProgramToInfer.input_range, (no_of_elements_input, 1))

        no_of_testcases = 100
        MRs_types = os.listdir(f"{MRs_path}/phase2")

        index_mutant = 0
        while index_mutant < number_of_mutants:
            stats.loc[f"{func_index}_{index_mutant}", "func_index"] = func_index
            stats.loc[f"{func_index}_{index_mutant}", "index_mutant"] = index_mutant

            for MRs_type in MRs_types:
                # for the MRs stored in npz format
                if MRs_type.startswith(f"{func_index}_") and MRs_type.endswith(".npz"):
                    parameters = MRs_type[-26:-17]
                    parameters_int = [int(e) for e in parameters.split("_")]
                    no_of_inputs = parameters_int[0]
                    mode_input_relation = parameters_int[1]
                    mode_output_relation = parameters_int[2]
                    degree_of_input_relation = parameters_int[3]
                    degree_of_output_relation = parameters_int[4]

                    # For each type of MR, check AB one by one, if fail to pass then mark isKill to True
                    isKill = 0
                    candidates_all = np.load(f"{MRs_path}/phase2/{MRs_type}")
                    A_candidates = candidates_all["A_candidates"]
                    B_candidates = candidates_all["B_candidates"]

                    all_count = A_candidates.shape[0]
                    for index_candidate in range(all_count):
                        A = A_candidates[index_candidate]
                        B = B_candidates[index_candidate]

                        i0_all = Phase1_PSOSearch.generate_i0_all(inputcases_range, no_of_testcases, no_of_elements_input)
                        survive_cost = Phase2_Filter.get_cost_of_AB(program, index_mutant, A, B, i0_all,
                                                                    mode_input_relation,
                                                                    mode_output_relation, degree_of_input_relation,
                                                                    degree_of_output_relation,
                                                                    no_of_elements_output)

                        if survive_cost > 0.05:
                            isKill = 1
                            print("killed by:")
                            print(A)
                            print(B)
                            break
                    stats.loc[f"{func_index}_{index_mutant}", parameters] = isKill
                    print(f"func_index is {func_index}, index_mutant is {index_mutant}, parameters is {parameters}, iskill is {isKill}")


            # for the MRs stored in pkl format
                elif MRs_type.startswith(f"{func_index}_") and MRs_type.endswith(".pkl"):
                    i0_all = Phase1_PSOSearch.generate_i0_all(inputcases_range, no_of_testcases, no_of_elements_input)

                    with open(f"{MRs_path}/phase2/{MRs_type}", "rb") as f:
                        MRs_dict = pickle.load(f)
                    for parameters, MRs in MRs_dict.items():
                        # print(f"func_index is {func_index}, parameters = {parameters}")
                        if parameters[0] == "x":
                            x_all_dict = MRs[0]
                            # print(x_all_dict)
                            y_all_df = MRs[1]
                            hDIR = MRs[2]
                            y_isKill_df = pd.DataFrame()

                            for index_i0 in range(i0_all.shape[0]):
                                i0 = i0_all[index_i0]
                                u = Phase1_PSOSearch.comb(i0, hDIR)
                                x_value_dict = {}
                                y_element_value_dict = {}
                                for x_name, A in x_all_dict.items():
                                    # print(f"x_name is {x_name}")
                                    # print(f"A is {A}")
                                    x = np.dot(A, u)
                                    x_value_dict[x_name] = x
                                    y = program(x, index_mutant)
                                    for index_eo in range(no_of_elements_output):
                                        y_element_value_dict[f"f{x_name}_{index_eo + 1}"] = y[index_eo]
                                y0 = program(i0, index_mutant)
                                for index_eo in range(no_of_elements_output):
                                    y_element_value_dict[f"fx0_{index_eo + 1}"] = y0[index_eo]

                                y_all_names = y_all_df.columns.values
                                y_all_values = np.zeros(y_all_names.shape)
                                for index_y in range(y_all_names.shape[0]):
                                    y_names = list(y_all_names[index_y])
                                    y_elements = []
                                    for ii in range(len(y_names)):
                                        try:
                                            y_elements.append(float(y_names[ii]))
                                        except:
                                            y_elements.append(y_element_value_dict[y_names[ii]])
                                    y_all_values[index_y] = np.product(y_elements)

                                for index_MR in range(y_all_df.shape[0]):
                                    B = y_all_df.iloc[index_MR, :].values
                                    Bv = np.dot(B, y_all_values)
                                    if np.isreal(Bv) and not np.isnan(Bv):
                                        if np.abs(Bv) < 0.1:
                                            y_isKill_df.loc[index_MR, index_i0] = 0
                                        else:
                                            y_isKill_df.loc[index_MR, index_i0] = 1
                                    else:
                                        y_isKill_df.loc[index_MR, index_i0] = 1

                            isKill = 0
                            for index_MR in range(y_isKill_df.shape[0]):
                                kill_number = np.sum(y_isKill_df.iloc[index_MR, :].values)
                                cost = np.divide(kill_number, no_of_testcases)
                                if cost >= 0.05:
                                    isKill = 1
                                    break

                            stats.loc[f"{func_index}_{index_mutant}", parameters] = isKill
                            print(f"func_index is {func_index}, index_mutant is {index_mutant}, parameters is {parameters}, iskill is {isKill}")

                        else:
                            parameters_int = [int(e) for e in parameters.split("_")]
                            no_of_inputs = parameters_int[0]
                            mode_input_relation = parameters_int[1]
                            mode_output_relation = parameters_int[2]
                            degree_of_input_relation = parameters_int[3]
                            degree_of_output_relation = parameters_int[4]

                            x_all_dict = MRs[0]
                            y_all_df = MRs[1]
                            y_isKill_df = pd.DataFrame()

                            for index_i0 in range(i0_all.shape[0]):
                                i0 = i0_all[index_i0]
                                u = Phase1_PSOSearch.comb(i0, degree_of_input_relation)
                                # print(u)
                                x_value_dict = {}
                                y_element_value_dict = {}
                                for x_name, A in x_all_dict.items():
                                    # print(x_name)
                                    # print(A)
                                    x = np.dot(A, u)
                                    x_value_dict[x_name] = x
                                    y = program(x, index_mutant)
                                    for index_eo in range(no_of_elements_output):
                                        y_element_value_dict[f"f{x_name}_{index_eo + 1}"] = y[index_eo]
                                y0 = program(i0, index_mutant)
                                for index_eo in range(no_of_elements_output):
                                    y_element_value_dict[f"fx0_{index_eo + 1}"] = y0[index_eo]

                                y_all_names = y_all_df.columns.values
                                y_all_values = np.zeros(y_all_names.shape)
                                for index_y in range(y_all_names.shape[0]):
                                    y_names = list(y_all_names[index_y])
                                    y_elements = []
                                    for ii in range(len(y_names)):
                                        try:
                                            y_elements.append(float(y_names[ii]))
                                        except:
                                            y_elements.append(y_element_value_dict[y_names[ii]])
                                    y_all_values[index_y] = np.product(y_elements)

                                for index_MR in range(y_all_df.shape[0]):
                                    B = y_all_df.iloc[index_MR, :].values
                                    Bv = np.dot(B, y_all_values)
                                    if np.isreal(Bv) and not np.isnan(Bv):
                                        if np.abs(Bv) < 0.1:
                                            y_isKill_df.loc[index_MR, index_i0] = 0
                                        else:
                                            y_isKill_df.loc[index_MR, index_i0] = 1
                                    else:
                                        y_isKill_df.loc[index_MR, index_i0] = 1

                            isKill = 0
                            for index_MR in range(y_isKill_df.shape[0]):
                                kill_number = np.sum(y_isKill_df.iloc[index_MR, :].values)
                                cost = np.divide(kill_number, no_of_testcases)
                                if cost >= 0.05:
                                    isKill = 1
                                    break

                            stats.loc[f"{func_index}_{index_mutant}", parameters] = isKill
                            print(f"func_index is {func_index}, index_mutant is {index_mutant}, parameters is {parameters}, iskill is {isKill}")


            index_mutant += 1

        stats.to_csv(f"{MRs_path}/temp_{func_index}_phase2_mutants.csv")


if __name__ == '__main__':
    MRs_path = ProgramToTest.MRs_path
    func_indices = ProgramToTest.func_indices
    main()
