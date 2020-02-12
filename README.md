ARSG is an efficient method for training neural networks. The acronym is derived from an adaptive remote stochastic gradient method. In training logistic regression on MNIST and Resnet-20 on CIFAR10 with fixed optimal hyper-parameters obtained by grid search, ARSG roughly halves the computation compared with ADAM. In the experiments, it also outperforms RANGER, which is a promising adaptive method proposed very recently by combining RADAM and the lookahead optimizer. 

The paper is available at https://arxiv.org/abs/1905.01422. 

We present a convergence proof for non-convex problems (in Nonconvex.pdf), which is more applicable for training deep neural
works. The results show that generalized ARSG shares the form of the convergence bound of the generalized ADAM methods, and
improves the coefficients. Particularly, ARSG with the preconditioner of AMSGRAD also shares the $O(log(T)/\sqrt{T}$ convergence rate of AMSGRAD, whilst the coefficients are improved.

NAMSG is a former name of ARSG.
