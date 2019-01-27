import os
import numpy as np
import pandas as pd
import datetime

import ProgramToInfer
import Phase1_PSOSearch
import sys


# the cost function for one particle (a pair of A and B)
def get_cost_of_AB(program_name, func_index, A, B, i0_all, mode_input_relation, mode_output_relation,
                   degree_of_input_relation,
                   degree_of_output_relation, no_of_elements_output):
    if mode_output_relation == 1:
        cost_of_AB = 1.0

        for index_i0 in range(i0_all.shape[0]):
            i0 = i0_all[index_i0]
            comb_i0 = Phase1_PSOSearch.comb(i0, degree_of_input_relation)
            i = Phase1_PSOSearch.generate_i(i0, comb_i0, A, mode_input_relation)
            o = Phase1_PSOSearch.get_o(program_name, func_index, i, no_of_elements_output)
            o_flatten = np.ravel(o)
            comb_o = Phase1_PSOSearch.comb(o_flatten, degree_of_output_relation)

            distance = np.dot(B, comb_o)
            if np.isreal(distance) and not np.isnan(distance):
                if np.abs(distance) < 0.1:
                    cost_of_AB -= 1.0 / i0_all.shape[0]


    elif mode_output_relation == 2:
        cost_of_AB = 1.0

        for index_i0 in range(i0_all.shape[0]):
            i0 = i0_all[index_i0]
            comb_i0 = Phase1_PSOSearch.comb(i0, degree_of_input_relation)
            i = Phase1_PSOSearch.generate_i(i0, comb_i0, A, mode_input_relation)
            o = Phase1_PSOSearch.get_o(program_name, func_index, i, no_of_elements_output)
            o_flatten = np.ravel(o)
            comb_o = Phase1_PSOSearch.comb(o_flatten, degree_of_output_relation)

            distance = np.dot(B, comb_o)
            if np.isreal(distance) and not np.isnan(distance):
                if distance > 0:
                    cost_of_AB -= 1.0 / i0_all.shape[0]

    elif mode_output_relation == 3:
        cost_of_AB = 1.0

        for index_i0 in range(i0_all.shape[0]):
            i0 = i0_all[index_i0]
            comb_i0 = Phase1_PSOSearch.comb(i0, degree_of_input_relation)
            i = Phase1_PSOSearch.generate_i(i0, comb_i0, A, mode_input_relation)
            o = Phase1_PSOSearch.get_o(program_name, func_index, i, no_of_elements_output)
            o_flatten = np.ravel(o)
            comb_o = Phase1_PSOSearch.comb(o_flatten, degree_of_output_relation)

            distance = np.dot(B, comb_o)
            if np.isreal(distance) and not np.isnan(distance):
                if distance < 0:
                    cost_of_AB -= 1.0 / i0_all.shape[0]

    return cost_of_AB

def main():
    no_of_inputcases = 100
    file_statistics = pd.DataFrame()

    parameters_int = [int(e) for e in parameters.split("_")]
    no_of_inputs = parameters_int[0]
    mode_input_relation = parameters_int[1]
    mode_output_relation = parameters_int[2]
    degree_of_input_relation = parameters_int[3]
    degree_of_output_relation = parameters_int[4]

    no_of_elements_input = ProgramToInfer.getNEI(func_index)
    no_of_elements_output = ProgramToInfer.getNEO(func_index)

    inputcases_range = ProgramToInfer.get_input_range(func_index)

    A_candidates_after_filter = []
    B_candidates_after_filter = []
    ini_count = 0
    survive_count = 0

    results_all = np.load('{}/phase1/{}'.format(output_path, output_name))
    min_cost_candidates = results_all['min_cost_candidates']
    A_candidates = results_all['A_candidates']
    B_candidates = results_all['B_candidates']
    all_count = min_cost_candidates.shape[0]

    for index_candidate in range(all_count):
        min_cost = min_cost_candidates[index_candidate]
        A = A_candidates[index_candidate]
        B = B_candidates[index_candidate]

        isPass = True
        isPassPhase1 = False

        if mode_output_relation == 1:
            if min_cost < 10.0:
                ini_count += 1
                isPassPhase1 = True
        else:
            if min_cost < 0.05:
                ini_count += 1
                isPassPhase1 = True

        if isPassPhase1:
            for index_test in range(100):
                i0_all = Phase1_PSOSearch.generate_i0_all(inputcases_range, no_of_inputcases, no_of_elements_input)
                survive_cost = get_cost_of_AB(ProgramToInfer.program, func_index, A, B, i0_all,
                                              mode_input_relation,
                                              mode_output_relation, degree_of_input_relation,
                                              degree_of_output_relation,
                                              no_of_elements_output)
                if survive_cost >= 0.05:
                    isPass = False
                    break

            if isPass:
                survive_count += 1
                A_candidates_after_filter.append(A)
                B_candidates_after_filter.append(B)

    results_all.close()

    A_candidates_after_filter = np.array(A_candidates_after_filter)
    B_candidates_after_filter = np.array(B_candidates_after_filter)

    if not os.path.isdir("{}/phase2".format(output_path)):
        os.mkdir("{}/phase2".format(output_path))

    np.savez(f'{output_path}/phase2/{func_index}_{parameters}_after_filter.npz', A_candidates=A_candidates_after_filter, B_candidates=B_candidates_after_filter)

    file_statistics.loc[f"{func_index}_{parameters}", "all_count"] = all_count
    file_statistics.loc[f"{func_index}_{parameters}", "ini_count"] = ini_count
    file_statistics.loc[f"{func_index}_{parameters}", "survive_count"] = survive_count

    file_statistics.to_csv(f"{output_path}/{func_index}_results.csv")

    print(f"\n----------")
    print(f"file is {output_name}")
    print(f"func_index is {func_index}, parameters is {parameters}")
    print(f"all count is {all_count}, ini count is {ini_count}, survive count is {survive_count}")


if __name__ == '__main__':
    func_indices = [int(sys.argv[1])]

    output_path = ProgramToInfer.output_path

    output_names = os.listdir(f"{output_path}/phase1")

    for func_index in func_indices:
        print(func_index)
        times = pd.DataFrame()
        for output_name in output_names:
            if output_name.startswith(f"{func_index}_") and output_name.endswith(".npz"):
                print(output_name)
                # exapmle: 21_2_1_1_1_1.npz
                parameters = output_name[-13:-4]

                t1 = datetime.datetime.now()
                main()
                t2 = datetime.datetime.now()
                cost_time = np.round((t2-t1).total_seconds(), 2)

                times.loc[f"{func_index}_{parameters}", "filter"] = cost_time

        times.to_csv(f"{output_path}/{func_index}_times.csv")