import os
import pickle
import numpy as np
import pandas as pd
import ProgramToInfer

if __name__ == '__main__':
    print("=====Start Inferring=====")
    os.system("python Phase1_PSOSearch.py")
    os.system("python Phase2_Filter.py")
    os.system("python Phase3_RemoveRedundancy.py")
    print("=====Results=====")

    MRs_path = ProgramToInfer.output_path
    func_indices = ProgramToInfer.func_indices
    MRs_types = os.listdir(f"{MRs_path}/phase3")

    for func_index in func_indices:
        for MRs_type in MRs_types:
            # for MRs stored in npz format
            if MRs_type.startswith(f"{func_index}_") and MRs_type.endswith(".npz"):
                filer_phase2 = 0
                parameters = MRs_type[-26:-17]

                candidates_all = np.load(f"{MRs_path}/phase3/{MRs_type}")
                A_candidates = candidates_all["A_candidates"]
                B_candidates = candidates_all["B_candidates"]
                print("----------")
                for index in range(len(A_candidates)):
                    print("    ----------")
                    print(f"    func_index is {func_index}, NOI_MIR_MOR_DIR_DOR is {parameters}:")
                    print(f"    MR candidate {index+1}:")
                    print(f"    A is {A_candidates[index]}")
                    print(f"    B is {B_candidates[index]}")

            # for the MRs stored in pkl format
            elif MRs_type.startswith(f"{func_index}_MRs_each") and MRs_type.endswith(".pkl"):
                with open(f"{MRs_path}/phase3/{MRs_type}", "rb") as f:
                    MRs_dict = pickle.load(f)
                for parameters, MRs in MRs_dict.items():
                    print("----------")
                    print("    ----------")
                    print(f"    func_index is {func_index}, NOI_MIR_MOR_DIR_DOR is {parameters}:")
                    if parameters[0] == "x":
                        x_all_dict = MRs[0]
                        y_all_df = MRs[1]
                        hDIR = MRs[2]
                    else:
                        x_all_dict = MRs[0]
                        print(f"    input relation: (each row denotes the coefficients for constructing this input from the base input. Please see the paper for details.)")
                        for key, value in x_all_dict.items():
                            print(f"    {key}: {value}")
                        y_all_df = MRs[1]
                        consts = y_all_df["1"]
                        coeffs = y_all_df.drop("1", axis=1)
                        y_all_df = pd.concat([consts, coeffs], axis=1)
                        print(f"    output relation: (each row represents one MR; column named '1' denotes the constant term; column named 'fx0_1' denotes the coefficient for the first element of output of x0; column named 'fx0_2' (if output has more than one elements) denotes the coefficient for the second element of output of x0; column named 'fx1_1' denotes the coefficient for the first element of output of x1, etc.)")
                        print(f"{y_all_df}")
