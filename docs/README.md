TODO: this document is not complete yet

# Introduction
In this study, it has been investigated the relationship between dropout and the overfit for MNIST dataset. Dropout is an effective regulatization technique to prevent or at least reduce the overfit in neural network systems.

Here total 10 experiments have been carried out to calculate the effect of dropout for different kind of dropout implementations. The following 10 experiments were performed:
- No dropout 
- Dropout with constant value (0.5, 0.7, 0.9 keep rates respectively)
- Linear increasing keep rate with formula keep_rate = param * iteration
- Linear decreasing keep rate with formula keep_rate = 1.0 - param * iteration
- Convex increasing keep rate with formula keep_rate = param * iteration * iteration
- Convex decreasing keep rate with formula keep_rate = 1.0 - param * iteration * iteration
- Concave increasing keep rate with formula keep_rate = param * sqrt(iteration)
- Concave decreasing keep rate with formula keep_rate = 1.0 - param * sqrt(iteration)
- Semi Convex increasing keep rate with formula keep_rate = param * sqrt(iteration). Here semi means: First no dropout is applied for "n" iterations. Then concave dropout rate is applied for the remaining iterations.
- Semi Convex decreasing keep rate with formula keep_rate = 1.0 - param * sqrt(iteration)
- Semi Concave increasing keep rate with formula keep_rate = param * sqrt(iteration). 
- Semi Concave decreasing keep rate with formula keep_rate = 1.0 - param * sqrt(iteration)

To see the overfit clearly, following MNIST dataset sizes have been examined: 400, 800, 1000, 5000, 10000

# Experiments
All the scenarios run for the all the dataset sizes and the overfit value for each run is stored in a file to plot and see the calculate value.

And at each iteration validation and training error value is calculated.

# Results

The legend labels are:
- NO-DROPOUT: Do not apply dropout
- C05: Constant keeping rate 0.5 for all the iterations
- L055-0.95: Increasing linear dropout starting with starting keeping rate from 0.55 to 0.95
- L055-0.95: Increasing linear dropout starting with starting keeping rate from 0.55 to 0.95
- Concave055-0.95: Increasing concave dropout starting with starting keeping rate from 0.55 to 0.95
- Convex055-0.95: Increasing convex dropout starting with starting keeping rate from 0.55 to 0.95
- Concave095-0.55: Decreasing concave dropout starting with starting keeping rate from 0.95 to 0.55
- Convex05-0.55: Decreasing convex dropout starting with starting keeping rate from 0.95 to 0.55
- HConcave055-0.95: No dropout for the first 25% iterations. Increasing concave dropout starting with starting keeping rate from 0.55 to 0.95
- HConvex055-0.95: No dropout for the first %25 iterations. Increasing convex dropout starting with starting keeping rate from 0.55 to 0.95
- HConcave095-0.55: No dropout for the first 25% iterations. Decreasing concave dropout starting with starting keeping rate from 0.95 to 0.55
- HConvex05-0.55: No dropout for the first 25% iterations. Decreasing convex dropout starting with starting keeping rate from 0.95 to 0.55

![MNIST experiments](overfit_400_10000.png?raw=true "Mnist experiments")

It has been ovserved that, "no-dropout" gives the most performance in terms of overfit with the highest overfit value for all the dataset sizes. Interestingly the constant dropout with the 0.5 gives best performance, but for the dataset size 10000, the lowest overfit value came from "convex-increasing" scenario. For further details, those experiments should be run for more dataset sizes such as 20k to 60k.

The following pictures give overfit values for each "category". The experiments are also divided into following "categories":
- DEC: Decreasing
- INC: Increasing
- CONSTANT: Constant
- HALF: No dropout for the half of the iterations, then apply dropout
- NO: No dropout


![Constant dropout MNIST](overfit_constant_400_10000.png?raw=true "Constant category")

![Decreasing dropout MNIST](overfit_dec_400_10000.png?raw=true "Decreasing category")

![Increasing dropout MNIST](overfit_inc_400_10000.png?raw=true "Increasing category")

![Half dropout MNIST](overfit_half_400_10000.png?raw=true "Half category")

![No dropout MNIST](overfit_no_400_10000.png?raw=true "Constant category")

# References
MNIST Datset: [LeCun et al., 1998a]
Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998. [on-line version]


