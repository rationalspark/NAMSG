#How to use the ARSG optimizer

1. Move arsg.py to the optim directory of PyTorch.
2. Modify \_\_init\_\_.py adding 
  from .arsg import Arsg
  del arsg
  
We also present an implementation of RANGER (https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer) for comparison, where the weight decay is modified to make it consistent with the PyTorch offical implementation of ADAM. The optimizer can be used by
1. Move ranger.py to the optim directory of PyTorch.
2. Modify \_\_init\_\_.py adding 
  from .ranger import Ranger
  del ranger
