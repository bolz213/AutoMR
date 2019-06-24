# AutoMR
Help to discover polynomial metamorphic relations (MRs).

_A Simple Example:_ To calculate ___sin(x)___, the program can be implemented as series:
```
sin(x) ≈ x - (x^3)/3! + (x^5)/5! - (x^7)/7! ...
```

It's difficult to find some properties of this program such as its relation with 3.1415926 (pi). However, AutoMR could find such interesting MRs: 
- sin(x) + sin(-x) = 0
- sin(x) + sin(x + 3.14159265) ≈ 0
- sin(x) - sin(x + 6.28318531) ≈ 0
- ...

_Another Example:_ For the program ___max(*args)___, AutoMR could find such MRs: 
- ___max(i<sup>(1)</sup>, i<sup>(2)</sup>, i<sup>(3)</sup>) - max(i<sup>(2)</sup>, i<sup>(1)</sup>, i<sup>(3)</sup>) = 0___
- ___max(i<sup>(1)</sup>-1, i<sup>(2)</sup>-2, i<sup>(3)</sup>-2) - max(i<sup>(1)</sup>, i<sup>(2)</sup>, i<sup>(3)</sup>) < 0___
- ...

## How to use AutoMR

### 0. Environemt settings
Tested on Windows 10 1809, Ubuntu 18.04, macOS Mojave with the following packages:
* python 3.6.3
* numpy 1.14.5
* pandas 0.23.1
* sympy 1.1.1
* scipy 1.1.0
* z3-solver 4.8.5.0
* scikit-learn 0.19.2

[Anaconda](https://www.anaconda.com/what-is-anaconda) is recommended to set up the environment.
With Anaconda, the dependencies can be easily installed by: 
```aidl
>>> cd path/to/AutoMR
>>> conda env create -f environment.yml
```

A virtual envoronement named "AutoMR" will be created together with the required dependencies. The following cmmand will activate the virtual environment named "AutoMR":
```
>>> conda activate AutoMR
```


### 1. Subject Programs

The information of the program which you want to infer MRs from should be configured in _ProgramToInfer.py_
> 1. Encapsulate the program in _program(i, func_index)_. _i_ is an array containing all the values to be passed to the program. _func_index_ is the index assigned to the program, which can facilitate inferring MRs for various programs in a batch.
> 2. Provide the number of elements of the input of the program in _getNEI(func_index)_.
> 3. Provide the number of elements of the output of the program in _getNEO(func_index)_.
> 4. Provide the input domain in _get_input_range(func_index)_.
> 5. Provide the input data type in _get_input_datatype(func_index)_.
> 6. Set the type of MRs you want to infer in _parameters_collection_. Each type is represented by a string consisting of "NOI_MIR_MOR_DIR_DOR". NOI is number of inputs involved in the target MR. MIR is mode of the input relation. MOR is mode of the output relation. For MIR and MOR, 1 means equality, 2 means greater-than, 3 means less-than. DIR and DOR is the polynomial degree of the input and output relations: 1 is linear, 2 is quadratic, etc.
> 7. Set the path to store the results in _output_path_.
> 8. Set the number of searches in _pso_runs_.


### 2. Search for MRs (Phase 1), filter the search results (Phase 2) and remove redundant MRs (Phase 3)
After setting up the subject programs and parameters, execute the following command. The results of each phase will be stored in corresponding folder under the _output_path_.
```
>>> python main.py
```

## A minimal example
The ___ProgramToInfer.py___ has already set up for inferring MRs for __sine__ program. You can just clone this repo and run `python main.py`, then after searching the results will be shown.

The following code block shows an example run and the results. The search was run 3 times and after filtering and redundancy removal 2 MRs were kept. New inputs could be constructed as: `x1 = 9.42 + x0, x2 = 6.28 + x0, x3 = 9.42 - x0`. The two MRs were: `1.732sin(x0) + 0.577sin(x1) - 0.577sin(x2) - 0.577sin(x3) ≈ 0` and `0.707sin(x2) - 0.707sin(x3) ≈ 0`

```
>>> python main.py
=====Start Inferring=====
start phase1: searching for MRs...
    searching: func_index is 1, parameters is 2_1_1_1_1, pso_run is 1.
    searching: func_index is 1, parameters is 2_1_1_1_1, pso_run is 2.
    searching: func_index is 1, parameters is 2_1_1_1_1, pso_run is 3.
----------
start phase2: filtering...
----------
start phase3: removing redundancy...
done
=====Results=====
----------
    ----------
    func_index is 1, NOI_MIR_MOR_DIR_DOR is 2_1_1_1_1:
    input relation A: (each row denotes the coefficients for constructing this input from the base input. Please see the paper for details.)
    x1: [[9.42477796 1.        ]]
    x2: [[6.28318531 1.        ]]
    x3: [[ 9.42477796 -1.        ]]
    output relation B: (each row represents one MR; column named '1' denotes the constant term; column named 'fx0_1' denotes the coefficient for the first element of output of x0; column named 'fx0_2' (if output has more than one elements) denotes the coefficient f
or the second element of output of x0; column named 'fx1_1' denotes the coefficient for the first element of output of x1, etc.)
              1      (fx0_1,)  (fx1_1,)  (fx2_1,)  (fx3_1,)
0  2.101669e-17  1.732051e+00   0.57735 -0.577350 -0.577350
1  7.408425e-17  1.110223e-16   0.00000  0.707107 -0.707107
```

### Paper data
The folder "PaperResults" contain the results of inferred MRs from a number of NumPy and Apache Math programs.
