# AutoMR
Help to discover polynomial metamorphic relations (MRs).

## A very simple example
For the program ___max(*args)___, AutoMR will help you find that 
- ___max(x<sup>(1)</sup>, x<sup>(2)</sup>, x<sup>(3)</sup>) - max(x<sup>(2)</sup>, x<sup>(1)</sup>, x<sup>(3)</sup>) = 0___
 - ___max(x<sup>(1)</sup>-1, x<sup>(2)</sup>-2, x<sup>(3)</sup>-3) - max(x<sup>(1)</sup>, x<sup>(2)</sup>, x<sup>(3)</sup>) < 0___
 - ...

## How to use AutoMR

### set parameters

The information of the program which you want to infer MRs from should be set in ___ProgramToInfer.py___
> 1. Encapsulate the program in _program(i, func_index)_. _i_ is an array containing all the values to be passed to the program. _func_index_ is the index assigned to the program, which is used when inferring MRs for various programs in a batch.
> 2. Set the number of elements of the input of the program in _getNEI(func_index)_.
> 3. Set the number of elements of the output of the program in _getNEO(func_index)_.
> 4. Set the indices of the programs which you want to infer MRs from in _func_indices_.
> 5. Set the type of MRs you want to infer in _parameters_collection_. Each type is represented by a string consisting of "NOI_MIR_MOR_DIR_DOR". NOI is number of inputs involved in the type of MR. MIR is mode of the input relation. MOR is mode of the output relation. For MIR and MOR, 1 means equality, 2 means greaterthan, 3 means lessthan. DIR and DOR is the polynomial degree of the input and output relations. 1 is linear, 2 is quadratic, etc.
> 6. Set the path to store the results in _output_path_.
> 7. Set the number of searches in _pso_runs_, and iterations of each search in _pso_iterations_.
> 8. Set the input domain in _get_input_range(func_index)_.
> 9. Set the input data type in _get_input_datatype(func_index)_.

### search (phase1), filter (phase2) and remove redundancy (phase3)
Execute the following three commands one by one. The results of each phase will be corresponding folder under the _output_path_.


```
    > python Phase1_PSOSearch.py
    > python Phase2_Filter.py
    > python Phase3_RemoveRedundancy.py
```

Tested on:
* python 3.6.3
* numpy 1.14.5
* pandas 0.23.1
* sympy 1.1.1
* scipy 1.1.0
* z3-solver 4.5.1.0.post2
* scikit-learn 0.19.2

## If use the inferred MRs for testing (Each of the tested programs is called a mutant)

#### set the parameters
Set the parameters in ___ProgramToTest.py___.
> 1. For the program to be tested, set the _func_index_ of the corresponding original program.
> 2. Set the path to the MRs in _MRs_path_.
> 3. Set the number of mutants to be tested in _number_of_mutants_.
> 4. For each mutant, encapsulate the mutant in _get_mutant(func_index)_.

#### testing
Execute the following command:
```
    > python Testing.py
```
The testing results will be stored under the path as _testing.csv_.

 