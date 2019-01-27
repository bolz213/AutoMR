import os
import z3
import sys
import itertools
import pandas as pd
import numpy as np
import sympy as sp
from sklearn.decomposition import TruncatedSVD
import pickle
import datetime

import ProgramToInfer
import Phase1_PSOSearch

def str_comb(vector, highest_degree):
    comb_vector = ["1"]
    degree = 1
    while degree < highest_degree + 1:
        comb_vector_with_degree = [i for i in itertools.combinations_with_replacement(vector, degree)]
        comb_vector = comb_vector + comb_vector_with_degree
        degree += 1
    return comb_vector


def similarity(M1, M2, coeff_range, const_range, mutiple_modified=False):
    if not mutiple_modified:
        M1 = np.array(M1)
        M2 = np.array(M2)
    elif mutiple_modified:
        M1 = np.array(M1)
        M2 = np.array(M2)
        index_max = np.argmax(np.abs(M1))
        mutiple_factor = np.divide(M2[index_max], M1[index_max])
        M2 = np.multiply(M2, mutiple_factor)

    coeff_range = np.array(coeff_range)
    const_range = np.array(const_range)

    const_M1 = M1[..., 0:1]
    const_M2 = M2[..., 0:1]
    coeff_M1 = M1[..., 1:]
    coeff_M2 = M2[..., 1:]
    range_coeff = np.abs(coeff_range[1] - coeff_range[0])
    range_const = np.abs(const_range[1] - const_range[0])

    sim_const = np.divide(np.sqrt(np.sum(np.square(const_M1 - const_M2))), range_const)
    sim_coeff = np.divide(np.sqrt(np.sum(np.square(coeff_M1 - coeff_M2))), range_coeff)

    sim = np.divide(np.add(sim_coeff, sim_const), np.prod(M1.shape))
    return sim


# check redundancy when svd is not applicable
def isRedundant(no_of_ele_input, no_of_ele_output, no_of_inputs, mode_input_relation, mode_output_relation,
                degree_of_input_relation, degree_of_output_relation, A1, B1, A2, B2, coeff_range, const_range):
    similarity_thre = 0.05

    def isInputRd():
        if mode_input_relation == 1:
            if similarity(A1, A2, coeff_range, const_range) < similarity_thre:
                return True
            else:
                return False
        elif mode_input_relation in [2, 3]:
            if similarity(A1, A2, coeff_range, const_range) < similarity_thre:
                return True
            else:
                i0 = []
                i_element = 1
                while i_element <= no_of_ele_input:
                    x_i = z3.Real(f"x_{i_element}")
                    i0.append(x_i)
                    i_element += 1

                i0 = np.array(i0)
                comb_i0 = Phase1_PSOSearch.comb(i0, degree_of_input_relation)
                i1_to_end_A1 = np.dot(A1, comb_i0)
                i1_to_end_A2 = np.dot(A2, comb_i0)

                if mode_input_relation == 2:
                    isInputRd_all_1_to_end_inputs = []
                    for i_input in range(no_of_inputs - 1):
                        i1_to_end_A1_i_input = i1_to_end_A1[i_input]
                        i1_to_end_A2_i_input = i1_to_end_A2[i_input]

                        isInputRd_i_input = []
                        for i_element in range(no_of_ele_input):
                            s = z3.Solver()
                            s.add(i1_to_end_A1_i_input[i_element] > i1_to_end_A2_i_input[i_element])
                            if s.check() == z3.unsat:
                                isInputRd_i_input.append(1)
                            else:
                                return False
                        if np.product(isInputRd_i_input):
                            isInputRd_all_1_to_end_inputs.append(1)
                        else:
                            return False
                    if np.product(isInputRd_all_1_to_end_inputs):
                        return True
                    else:
                        return False

                elif mode_input_relation == 3:
                    isInputRd_all_1_to_end_inputs = []
                    for i_input in range(no_of_inputs - 1):
                        i1_to_end_A1_i_input = i1_to_end_A1[i_input]
                        i1_to_end_A2_i_input = i1_to_end_A2[i_input]

                        isInputRd_i_input = []
                        for i_element in range(no_of_ele_input):
                            s = z3.Solver()
                            s.add(i1_to_end_A1_i_input[i_element] < i1_to_end_A2_i_input[i_element])
                            if s.check() == z3.unsat:
                                isInputRd_i_input.append(1)
                            else:
                                return False
                        if np.product(isInputRd_i_input):
                            isInputRd_all_1_to_end_inputs.append(1)
                        else:
                            return False
                    if np.product(isInputRd_all_1_to_end_inputs):
                        return True
                    else:
                        return False

    def isOutputRd():
        if mode_output_relation == 1:
            if similarity(B1, B2, coeff_range, const_range, mutiple_modified=True) < similarity_thre:
                return True
            else:
                return False
        elif mode_output_relation in [2, 3]:
            if similarity(B1, B2, coeff_range, const_range, mutiple_modified=True) < similarity_thre:
                return True
            else:
                s2 = z3.Solver()

                outputs = []
                i = 1
                while i <= (no_of_ele_output * no_of_inputs):
                    outputs_i = z3.Real('y_{}'.format(i))
                    outputs.append(outputs_i)
                    i += 1

                outputs = np.array(outputs)
                y = Phase1_PSOSearch.comb(outputs, degree_of_output_relation)
                output_relation1 = np.dot(B1, y)
                output_relation2 = np.dot(B2, y)

                if mode_output_relation == 2:
                    s2.add(output_relation1 > output_relation2)
                    if s2.check() == z3.unsat:
                        return True
                    else:
                        return False

                elif mode_output_relation == 3:
                    s2.add(output_relation1 < output_relation2)
                    if s2.check() == z3.unsat:
                        return True
                    else:
                        return False

    if mode_input_relation == 1:
        if mode_output_relation == 1:
            return isInputRd()
        elif mode_output_relation in [2, 3]:
            return isInputRd() and isOutputRd()
    elif mode_input_relation in [2, 3]:
        return isInputRd() and isOutputRd()

    # print("-----A-----")
    # print(A1, A2)
    # print(isInputRd())
    # print("-----B-----")
    # print(B1, B2)
    # print(isOutputRd())


def deredundant(no_of_ele_input, no_of_ele_output, no_of_inputs, mode_input_relation, mode_output_relation,
                degree_of_input_relation, degree_of_output_relation, A_candidates, B_candidates, coeff_range, const_range):
    before_z3 = A_candidates.shape[0]

    size = A_candidates.shape[0]
    indices_Red = []

    index_base = 0
    while index_base < size:
        if index_base in indices_Red:
            index_base += 1
            continue
        else:

            index_compare = 0
            while index_compare < size:
                if (index_compare in indices_Red) or (index_compare == index_base):
                    index_compare += 1
                    continue
                else:
                    if isRedundant(no_of_ele_input, no_of_ele_output, no_of_inputs, mode_input_relation,
                                   mode_output_relation, degree_of_input_relation, degree_of_output_relation,
                                   A_candidates[index_base], B_candidates[index_base], A_candidates[index_compare],
                                   B_candidates[index_compare], coeff_range, const_range):
                        indices_Red.append(index_compare)
                index_compare += 1

        index_base += 1

    A_candidates_after_z3 = np.delete(A_candidates, indices_Red, axis=0)
    B_candidates_after_z3 = np.delete(B_candidates, indices_Red, axis=0)
    after_z3 = A_candidates_after_z3.shape[0]
    return A_candidates_after_z3, B_candidates_after_z3, before_z3, after_z3


# MRs should be a dict{parameters: [A, B]}
def svd_check(MRs, NEOI, NEOO):
    # get the highest NOI DIR DOR for MRs_equal_equal
    hNOI = 0
    hDIR = 0
    hDOR = 0
    for parameters in MRs.keys():
        parameters_int = [int(e) for e in parameters.split("_")]
        hNOI = max(hNOI, parameters_int[0])
        hDIR = max(hDIR, parameters_int[3])
        hDOR = max(hDOR, parameters_int[4])

    # to store all distinct As
    x_all = {}

    hu = str_comb([f"x0_{i+1}" for i in range(NEOI)], hDIR)

    # x0_1 = pd.DataFrame([[0, 1, 0]], index=["e1"], columns=u)
    # x_temp = x_temp.append(x0_1, ignore_index=False, sort=False)
    # x0_2 = pd.DataFrame([[0, 0, 1]], index=["e2"], columns=u)
    # x_temp = x_temp.append(x0_2, ignore_index=False, sort=False)
    # x_all["x0"] = x_temp.fillna(0).values
    # # print(x_all["x0"])

    MR_all = []
    for parameters, AB_after_CS in MRs.items():
        parameters_int = [int(e) for e in parameters.split("_")]
        NOI = parameters_int[0]
        MIR = parameters_int[1]
        MOR = parameters_int[2]
        DIR = parameters_int[3]
        DOR = parameters_int[4]

        As = AB_after_CS[0]
        Bs = AB_after_CS[1]

        u = str_comb([f"x0_{i+1}" for i in range(NEOI)], DIR)

        for i_A in range(As.shape[0]):
            A = As[i_A]

            # store the fx
            o_orig = ["fx0"]

            for i_NOI in range(A.shape[0]):
                A_iNOI = A[i_NOI]

                # check whether add a new x or not
                x_temp = pd.DataFrame(columns=hu)
                for i_EOI in range(NEOI):
                    x_temp_iEOI = pd.DataFrame([A_iNOI[i_EOI]], columns=u, index=[f"e{i_EOI+1}"])
                    x_temp = x_temp.append(x_temp_iEOI, ignore_index=False, sort=False)
                    x_temp = x_temp.fillna(0)
                # print(x_temp)
                isNew = True
                for x, A_x in x_all.items():
                    # print(A_x)
                    # print(x_temp.values)
                    isExist = np.allclose(A_x, x_temp.values,atol=0.05, rtol=0.1, equal_nan=True)
                    # print(isExist)
                    if isExist:
                        o_orig.append(f"f{x}")
                        isNew = False
                        break
                if isNew:
                    # print(len(x_all))
                    number_of_x = len(x_all)
                    x_all[f"x{number_of_x + 1}"] = x_temp.values
                    o_orig.append(f"fx{number_of_x + 1}")
                    # print(o_orig)

            # create corresponding output elemets
            o = []
            for i in range(len(o_orig)):
                for i_ele in range(NEOO):
                    o.append(f"{o_orig[i]}_{i_ele + 1}")

            # create v
            v = str_comb(o, DOR)
            # print(v)

            MR = pd.DataFrame([Bs[i_A]], columns=v)
            MR = MR.groupby(MR.columns, axis=1).sum()
            # print(MR.columns)
            MR_all.append(MR)

    # for i in range(len(MR_all)):
    #     print(MR_all[i].columns)

    MR_all_df = pd.concat((df for df in MR_all), ignore_index=True, sort=True)
    y_all = MR_all_df.columns
    # print(len(y_all))
    MR_all_df = MR_all_df.fillna(0)
    MR_all_array = MR_all_df.values
    # print(MR_all_array)

    # # filter by rref
    # MR_all_before_rref = sp.Matrix(MR_all_array)
    # MR_all_after_rref = np.array(MR_all_before_rref.rref()[0])
    # # print(MR_all_after_rref)
    # # print(MR_all_after_rref.shape)
    # indepentent_indices = [e for e in MR_all_before_rref.rref()[1]]
    #
    # if len(indepentent_indices) < min(MR_all_after_rref.shape):
    #     MR_all_array = MR_all_after_rref[indepentent_indices, ...]
    # else:
    #     pass

    # c = a.T
    # pca = PCA(n_components=3)
    # pca.fit(c)
    # c_pca = pca.transform(c)
    # print(np.round(pca.explained_variance_ratio_, 2))
    # a_pca = c_pca.T
    # print(a_pca)

    # filter by svd
    c = MR_all_array.T
    # print(c)
    svd = TruncatedSVD(n_components=(c.shape[1]-1))
    svd.fit(c)
    c_svd = svd.transform(c)
    c_after_svd = c_svd.T
    ratios = np.round(svd.explained_variance_ratio_, 2)

    MRs_matrix_after_svd = []
    accu_ration = 0

    for i in range(len(ratios)):
        MRs_matrix_after_svd.append(c_after_svd[i])
        accu_ration += ratios[i]

        if accu_ration > 0.99:
            break

    MRs_df_after_svd = pd.DataFrame(MRs_matrix_after_svd, columns=y_all)

    # print(y_all)
    return x_all, MRs_df_after_svd, hDIR


def main():
    if not os.path.isdir(f"{folder_path}/phase3"):
        os.mkdir(f"{folder_path}/phase3")

    results = pd.read_csv(f"{folder_path}/results.csv", index_col=0)

    times = pd.read_csv(f"{folder_path}/times.csv", index_col=0)

    for func_index in func_indices:
        stats_after_cs_svd_df = {}
        time_cs_svd = {}

        # get NEI NEO
        NEI = ProgramToInfer.getNEI(func_index)
        NEO = ProgramToInfer.getNEO(func_index)

        # select the filtered MRs of the func_name
        AB_all = {}
        for filename in os.listdir(f"{folder_path}/phase2"):
            if filename.startswith(f"{func_index}_") and filename.endswith("after_filter.npz"):
                # example: np_log1p_2_1_1_1_1_after_filter.npz
                parameters = filename[-26:-17]
                time_cs_svd[parameters] = 0
                MRs = np.load(f"{folder_path}/phase2/{filename}")
                As = MRs["A_candidates"]
                Bs = MRs["B_candidates"]
                if As.shape[0] > 1:
                    AB_all[parameters] = [As, Bs]
                elif As.shape[0] == 1:
                    stats_after_cs_svd_df[parameters] = As.shape[0]
                    np.savez(f'{folder_path}/phase3/{func_index}_{parameters}_after_cs_svd.npz', A_candidates=As, B_candidates=Bs)
                else:
                    stats_after_cs_svd_df[parameters] = As.shape[0]

        # filer using CS
        AB_all_after_CS = {}
        MRs_each_type_after_svd = {}
        for parameters, ABs in AB_all.items():
            t1 = datetime.datetime.now()

            parameters_int = [int(e) for e in parameters.split("_")]
            NOI = parameters_int[0]
            MIR = parameters_int[1]
            MOR = parameters_int[2]
            DIR = parameters_int[3]
            DOR = parameters_int[4]

            A_candidates = ABs[0]
            B_candidates = ABs[1]

            A_candidates_after_CS, B_candidates_after_CS, before_CS, after_CS = deredundant(NEI, NEO, NOI, MIR, MOR, DIR, DOR, A_candidates, B_candidates, coeff_range, const_range)
            AB_all_after_CS[parameters] = [A_candidates_after_CS, B_candidates_after_CS]

            # for output inequality MRs, can't use svd so just save cs results
            if parameters_int[2] != 1:
                stats_after_cs_svd_df[parameters] = A_candidates_after_CS.shape[0]
                np.savez(f'{folder_path}/phase3/{func_index}_{parameters}_after_cs_svd.npz', A_candidates=A_candidates_after_CS, B_candidates=B_candidates_after_CS)

            # do svd for output equality MRs
            else:
                # do svd for the type which has more than 1 MRs
                if A_candidates_after_CS.shape[0] > 1:
                    MRs_each_type_after_svd[parameters] = svd_check({parameters: AB_all_after_CS[parameters]}, NEI, NEO)
                    stats_after_cs_svd_df[parameters] = MRs_each_type_after_svd[parameters][1].shape[0]
                else:
                    stats_after_cs_svd_df[parameters] = A_candidates_after_CS.shape[0]
                    np.savez(f'{folder_path}/phase3/{func_index}_{parameters}_after_cs_svd.npz', A_candidates=A_candidates_after_CS, B_candidates=B_candidates_after_CS)

            t2 = datetime.datetime.now()
            cost_time = np.round((t2-t1).total_seconds(), 2)
            time_cs_svd[parameters] = time_cs_svd[parameters] + cost_time
        if len(MRs_each_type_after_svd) > 0:
            with open(f"{folder_path}/phase3/{func_index}_MRs_each_type_after_cs_svd.pkl", "wb") as f1:
                pickle.dump(MRs_each_type_after_svd, f1, pickle.HIGHEST_PROTOCOL)

        # for group of {equal input, equal output}, {greater, equal}, {less, equal}, use svd to simplify them
        MRs_equal_equal = {}
        MRs_greater_equal = {}
        MRs_less_equal = {}
        for parameters, candidates_after_CS in AB_all_after_CS.items():
            parameters_int = [int(e) for e in parameters.split("_")]
            if parameters_int[1] == 1 and parameters_int[2] == 1:
                MRs_equal_equal[parameters] = candidates_after_CS
            elif parameters_int[1] == 2 and parameters_int[2] == 1:
                MRs_greater_equal[parameters] = candidates_after_CS
            elif parameters_int[1] == 3 and parameters_int[2] == 1:
                MRs_less_equal[parameters] = candidates_after_CS
            else:
                pass

        MRs_group_after_svd = {}
        if len(MRs_equal_equal) > 0:
            t1 = datetime.datetime.now()
            MRs_group_after_svd["x_1_1_x_x"] = svd_check(MRs_equal_equal, NEI, NEO)
            t2 = datetime.datetime.now()
            cost_time = np.round((t2-t1).total_seconds(), 2)
            time_cs_svd["x_1_1_x_x"] = cost_time
            stats_after_cs_svd_df["x_1_1_x_x"] = MRs_group_after_svd["x_1_1_x_x"][1].shape[0]
        if len(MRs_greater_equal) > 0:
            t1 = datetime.datetime.now()
            MRs_group_after_svd["x_2_1_x_x"] = svd_check(MRs_greater_equal, NEI, NEO)
            t2 = datetime.datetime.now()
            cost_time = np.round((t2-t1).total_seconds(), 2)
            time_cs_svd["x_1_1_x_x"] = cost_time
            stats_after_cs_svd_df["x_2_1_x_x"] = MRs_group_after_svd["x_2_1_x_x"][1].shape[0]
        if len(MRs_less_equal) > 0:
            t1 = datetime.datetime.now()
            MRs_group_after_svd["x_3_1_x_x"] = svd_check(MRs_greater_equal, NEI, NEO)
            t2 = datetime.datetime.now()
            cost_time = np.round((t2-t1).total_seconds(), 2)
            time_cs_svd["x_1_1_x_x"] = cost_time
            stats_after_cs_svd_df["x_3_1_x_x"] = MRs_group_after_svd["x_3_1_x_x"][1].shape[0]

        if len(MRs_group_after_svd) > 0:
            with open(f"{folder_path}/phase3/{func_index}_MRs_group_after_cs_svd.pkl", "wb") as f2:
                pickle.dump(MRs_group_after_svd, f2, pickle.HIGHEST_PROTOCOL)

        # save number of MRs after svd
        # print(stats_after_cs_svd_df)
        for parameters, number in stats_after_cs_svd_df.items():
            results.loc[f"{func_index}_{parameters}", "after_cs_svd"] = number

        results.to_csv(f"{folder_path}/results.csv")

        for parameters, time in time_cs_svd.items():
            times.loc[f"{func_index}_{parameters}", "cs_svd"] = time

        times.to_csv(f"{folder_path}/times.csv")

if __name__ == '__main__':

    folder_path = ProgramToInfer.output_path
    func_indices = [int(sys.argv[1])]
    const_range = ProgramToInfer.const_range
    coeff_range = ProgramToInfer.coeff_range
    print("----------")
    print("removing redundancy...")
    main()
    print("done")
