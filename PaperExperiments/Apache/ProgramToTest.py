import numpy as np
import os

import jnius_config
jnius_config.set_classpath('.', './mutants/bin')

from jnius import autoclass

# number of programs to test
def get_number_of_mutants(func_index):
    if func_index == 26:
        return 18
    elif func_index == 27:
        return 8
    elif func_index == 28:
        return 298
    elif func_index == 29:
        return 95
    elif func_index == 30:
        return 20
    elif func_index == 31:
        return 116
    elif func_index == 32:
        return 59
    elif func_index == 33:
        return 19

# program to test
def get_program(func_index):
    def func21(i, index_mutant):
            if index_mutant == 0:
                Mu = autoclass("sin_mutants.sin_original")
                mu = Mu()
            else:
                Mu = autoclass(f"sin_mutants.sin_{index_mutant}")
                mu = Mu()
            i1 = np.float(i[0])
            return(np.array([mu.Sin(i1)]))

    def func1(i, index_mutant):
        if index_mutant == 0:
            Mu = autoclass(f"pku.abs")
            mu = Mu()
        else:
            mutants_dir = "./PSO_study2forASE/bin/abs_mutants"
            mutants = os.listdir(f"{mutants_dir}")
            try:
                mutants.remove(".DS_Store")
            except:
                pass
            Mu = autoclass(f"abs_mutants.{mutants[index_mutant - 1]}.abs")
            mu = Mu()

        i1 = i[0]
        o = np.array([mu.abs(i1)])
        return o

    def func5(i, index_mutant):
        if index_mutant == 0:
            Mu = autoclass(f"pku.asinh")
            mu = Mu()
        elif index_mutant >= 1:
            mutants_dir = "./PSO_study2forASE/bin/asinh_mutants"
            mutants = os.listdir(f"{mutants_dir}")
            try:
                mutants.remove(".DS_Store")
            except:
                pass
            Mu = autoclass(f"asinh_mutants.{mutants[index_mutant - 1]}.asinh")
            mu = Mu()

        i1 = i[0]
        o = np.array([mu.asinh(i1)])
        return o

    def func6(i, index_mutant):
        func_name = "atan"
        if index_mutant == 0:
            Mu = autoclass(f"pku.{func_name}")
            mu = Mu()
        elif index_mutant >= 1:
            mutants_dir = f"./PSO_study2forASE/bin/{func_name}_mutants"
            mutants = os.listdir(f"{mutants_dir}")
            try:
                mutants.remove(".DS_Store")
            except:
                pass
            Mu = autoclass(f"{func_name}_mutants.{mutants[index_mutant - 1]}.{func_name}")
            mu = Mu()

        i1 = i[0]
        o = np.array([mu.atan(i1)])
        return o

    def func10(i, index_mutant):
        if index_mutant == 0:
            Mu = autoclass("cos_mutants.cos_original")
            mu = Mu()
        elif index_mutant >= 1:
            Mu = autoclass(f"cos_mutants.cos_{index_mutant}")
            mu = Mu()

        i1 = i[0]
        o = np.array([mu.Cos(i1)])
        return o

    def func16(i, index_mutant):
        func_name = "log1p"
        if index_mutant == 0:
            Mu = autoclass(f"pku.{func_name}")
            mu = Mu()
        elif index_mutant >= 1:
            mutants_dir = f"./PSO_study2forASE/bin/{func_name}_mutants"
            mutants = os.listdir(f"{mutants_dir}")
            try:
                mutants.remove(".DS_Store")
            except:
                pass
            Mu = autoclass(f"{func_name}_mutants.{mutants[index_mutant - 1]}.{func_name}")
            mu = Mu()

        i1 = i[0]
        o = np.array([mu.log1p(i1)])
        return o

    def func17(i, index_mutant):
        func_name = "log10"
        if index_mutant == 0:
            Mu = autoclass(f"pku.{func_name}")
            mu = Mu()
        elif index_mutant >= 1:
            mutants_dir = f"./PSO_study2forASE/bin/{func_name}_mutants"
            mutants = os.listdir(f"{mutants_dir}")
            try:
                mutants.remove(".DS_Store")
            except:
                pass
            Mu = autoclass(f"{func_name}_mutants.{mutants[index_mutant - 1]}.{func_name}")
            mu = Mu()

        i1 = i[0]
        o = np.array([mu.log10(i1)])
        return o

    def func24(i, index_mutant):
        if index_mutant == 0:
            mu_Tan = autoclass(f"tan_mutants.tan_original")
            mu_tan = mu_Tan()
        elif index_mutant >= 1:
            mu_Tan = autoclass(f"tan_mutants.tan_{index_mutant}")
            mu_tan = mu_Tan()

        i1 = np.float(i[0])
        o = np.array([mu_tan.tan(i1)])
        return o

    if func_index == 26:
        return func21
    elif func_index == 27:
        return func1
    elif func_index == 28:
        return func5
    elif func_index == 29:
        return func6
    elif func_index == 30:
        return func10
    elif func_index == 31:
        return func16
    elif func_index == 32:
        return func17
    elif func_index == 33:
        return func24


