ARSG is an efficient method for training neural networks. The acronym is derived from the adaptive remote stochastic gradient method. ARSG yields $O(1/\sqrt{T})$ convergence rate in non-convex settings, that can be further improved to $O(\log(T)/T)$ in strongly convex settings. Numerical experiments demonstrate that ARSG achieves both faster convergence and better generalization, compared with popular adaptive methods, such as ADAM, NADAM, AMSGRAD, and RANGER for the tested problems. In training logistic regression on MNIST and Resnet-20 on CIFAR10 with fixed optimal hyper-parameters obtained by grid search, ARSG roughly halves the computation compared with ADAM. For training ResNet-50 on ImageNet, ARSG outperforms ADAM in convergence speed and meanwhile it surpasses SGD in generalization.

The paper is available at https://arxiv.org/abs/1905.01422. 

NAMSG is a former name of ARSG.
