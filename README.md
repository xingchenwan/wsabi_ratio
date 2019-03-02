# WSABI-Ratio

[Website](https://github.com/xingchenwan/wsabi_ratio) 

WSABI Ratio aims to apply the Fast Bayesian Quadrature framework [1] in the context of estimation of ratios between integrals. Such integrals are often encountered, for example, in evaluating the posterior distribution in Bayesian Inference and Markovian models.

This repository contains the codes for Xingchen Wan's Fourth-year project (4YP) for the degree of Master of Engineering in Engineering Science at University of Oxford (2019).

Main Contents

* **Synthetic Test Functions:** 1D and multidimensional Gaussian mixtures, 2D Rosenbrock function
* **Experiment on real data:** 
* GP Regression: Hydrodynamics dataset [2], Sotonmet [data](http://www.robots.ox.ac.uk/~mosb/teaching/AIMS_CDT/sotonmet.txt) 
* SVM Classification: Wisconsin Breast Cancer [data](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/)
* (New) Bayesian Changepoint Detection (BOCPD-GP): Nile dataset, Dow Jones Daily Log-return
* **Methods**: MLE/MAP, Grid Search (For SVM) Markov-chain Monte Carlo, Vanilla Bayesian Quadrature, Fast Warped Bayesian Quadrature

The codes have been written and tested under Anaconda Python 3.6. Other dependent packages include:
* GPy
* GPyOpt
* Amazon Emukit

Also used WSABI implementations written by Ed Wagstaff [3]

**References**


[1] Gunter, T., Osborne, M.A., Garnett, R., Hennig, P. and Roberts, S.J., 2014. Sampling for inference in probabilistic models with fast Bayesian quadrature. In Advances in neural information processing systems (pp. 2789-2797).
Vancouver	

[2] Gerritsma, J., Onnink, R. and Versluis, A., 1981. Geometry, resistance and stability of the delft systematic yacht hull series. International shipbuilding progress, 28(328), pp.276-297.


[3] Wagstaff, E., Hamid, S. and Osborne, M., 2018. Batch Selection for Parallelisation of Bayesian Quadrature. arXiv preprint arXiv:1812.01553.
Vancouver	


Updated 14 Feb 2019 | Xingchen Wan | xingchen.wan@st-annes.ox.ac.uk