# Transfer Component Analysis

This is the implementation of Transfer Component Analysis (TCA) in Python and Matlab.

Remark: The core of TCA is a generalized eigendecompsition problem. In Matlab, it can be solved by calling `eigs()` function. In Python, the implementation `scipy.linalg.eig()` function can do the same thing. However, they are a bit different. So the results may be different.

The test dataset can be downloaded [HERE](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/data).

**Reference**

Pan S J, Tsang I W, Kwok J T, et al. Domain adaptation via transfer component analysis[J]. IEEE Transactions on Neural Networks, 2011, 22(2): 199-210.