# AutoMR
Help to discover polynomial metamorphic relations (MRs).

_Example 1:_ For the program ___sin(x)___, it's usually implemented in math packages as series:
```
sin(x) ≈ x - (x^3)/3! + (x^5)/5! - (x^7)/7! ...
```

It's difficult to find some properties of this program such as its relation with pi. However, AutoMR could find such interesting MRs: 
- sin(x) + sin(-x) = 0
- sin(x) + sin(x + 3.14159265) ≈ 0
- sin(x) - sin(x + 6.28318531) ≈ 0
- ...

_Example 2:_ For the program ___max(*args)___, AutoMR could find such MRs: 
- ___max(i<sup>(1)</sup>, i<sup>(2)</sup>, i<sup>(3)</sup>) - max(i<sup>(2)</sup>, i<sup>(1)</sup>, i<sup>(3)</sup>) = 0___
- ___max(i<sup>(1)</sup>-1, i<sup>(2)</sup>-2, i<sup>(3)</sup>-2) - max(i<sup>(1)</sup>, i<sup>(2)</sup>, i<sup>(3)</sup>) < 0___
- ...

## How to use AutoMR

### 0. Environemt settings
Tested on:
* python 3.6.3
* numpy 1.14.5
* pandas 0.23.1
* sympy 1.1.1
* scipy 1.1.0
* z3-solver 4.5.1.0.post2
* scikit-learn 0.19.2

[Anaconda](https://www.anaconda.com/what-is-anaconda) is recommended to set up the environment.

### 1. Subject Programs

The information of the program which you want to infer MRs from should be set in ___ProgramToInfer.py___
> 1. Encapsulate the program in _program(i, func_index)_. _i_ is an array containing all the values to be passed to the program. _func_index_ is the index assigned to the program, which is used when inferring MRs for various programs in a batch.
> 2. Set the number of elements of the input of the program in _getNEI(func_index)_.
> 3. Set the number of elements of the output of the program in _getNEO(func_index)_.
> 5. Set the type of MRs you want to infer in _parameters_collection_. Each type is represented by a string consisting of "NOI_MIR_MOR_DIR_DOR". NOI is number of inputs involved in the target MR. MIR is mode of the input relation. MOR is mode of the output relation. For MIR and MOR, 1 means equality, 2 means greater-than, 3 means less-than. DIR and DOR is the polynomial degree of the input and output relations. 1 is linear, 2 is quadratic, etc.
> 6. Set the path to store the results in _output_path_.
> 7. Set the number of searches in _pso_runs_, and iterations within each pso_run in _pso_iterations_.
> 8. Set the input domain in _get_input_range(func_index)_.
> 9. Set the input data type in _get_input_datatype(func_index)_.

### 2. Search for MRs (Phase 1), filter the search results (Phase 2) and remove redundant MRs (Phase 3)
After setting up the subject programs and parameters, execute the following command. The results of each phase will be corresponding folder under the _output_path_.
```
> python main.py
```

### example
The ___ProgramToInfer.py___ has already set up for inferring MRs for __sine__ program. You can just clone this repo and run to see the results.

### Paper data
The folder "PaperResults" contain the results of inferred MRs from a number of NumPy and Apache Math programs.
