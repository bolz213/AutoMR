### This folder contains the inferred MRs for 29 programs in NumPy.

The MRs are grouped according to the parameters: FuncIndex_NoOfInputs_ModeOfInputRelation_ModeOfOutputRelation_DegreeOfInputRelation_DegreeOfOutputRelation.npz
> for example: 1_2_1_1_1_1.npz means:
>   - They are the MRs for the program with FuncIndex 1,
>   - The number of involved inputs is 2,
>   - The mode of input relation is 1 (1 means equality, 2 means greater than, 3 means less than),
>   - The mode of output relation is 2,
>   - The degree of the polynomial input relation is 1 (i.e., linear),
>   - The degree of the polynomial output relation is 1 (i.e., linear).

FuncIndex and its corresponding program:
> |FuncIndex|Program|
> |:----|:----|
> |np.abs|1|
> |np.acos|2|
> |np.acosh|3|
> |np.asin|4|
> |np.asinh|5|
> |np.atan|6|
> |np.atan2|7|
> |np.atanh|8|
> |np.ceil|9|
> |np.cos|10|
> |np.cosh|11|
> |np.exp|12|
> |np.floor|13|
> |np.hypot|14|
> |np.log|15|
> |np.log1p|16|
> |np.log10|17|
> |np.max|18|
> |np.min|19|
> |np.round|20|
> |np.sin|21|
> |np.sinh|22|
> |np.sqrt|23|
> |np.tan|24|
> |np.tanh|25|
> ||26|
> ||27|
> ||28|
> |np.sort|29|