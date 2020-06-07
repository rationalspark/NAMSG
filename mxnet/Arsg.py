#Add the class to optimizer.py in MXNET to use the Arsg optimizer

#Recommended hyper-parameters: beta1=0.999, beta2=0.99, mu=0.1, epsilon=1e-8,
#Use epsilon=1e-3 in the fine mode to improve generalization.
#learning_rate is obtained through grid search.


#The observation boost hyper-parameter (OBSB) policy
#Perform grid search for a small proportion of epoches to obtain the initial step size alpha0,with a beta1
#close to 1, and a relatively small mu. 
#In training, start with step size alpha0 and observation factor mu0. 
#Compute the average convergence rate at the interval of several epoches.
#Perform boosting when the loss flattens
#To apply the boosting, set mu=2*mu, and divide the step size roughly by 4. 
#OBSB is different from vanilla learning_rate decay, since a small mu enalbes large initial
#step size. After applying the observation boost, the step size is still relatively large.
#Consequently, we can still apply learning_rate decay to achieve acceleration.

#The recommend hyper-parameters for OBSB are  beta1=0.999, beta2=0.99, and mu=mu0=0.05. alpha should be roughly divided by 4 when $\mu$ is doubled.

#Use set_obs_fac_mu to set the observation factor mu.

@register
class Arsg(Optimizer):
    """
    The Arsg optimizer
    Parameters
    ----------
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional
        Small value to avoid dividing by 0.
    mu : float, optional
        Oberservation factor
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.999, beta2=0.99, epsilon=1e-8, mu=0.1, **kwargs):
        super(Arsg, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mu = mu

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context, dtype=weight.dtype),  # momentum
                zeros(weight.shape, weight.context, dtype=weight.dtype),  # v
                self.epsilon * ones(weight.shape, weight.context, dtype=weight.dtype))  # vMax
    
    #set the observation factor mu
    def set_obs_fac_mu(self,mu): 
        self.mu=mu

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        t = self._index_update_count[index]

        # Weight decay
        grad = grad * self.rescale_grad
        if wd !=0:
            grad = grad + wd * weight

        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        # update m, v, and vMax
        m_t, v_t, vMax_t = state
        m_t[:] = self.beta1 * m_t + (1. - self.beta1) * grad
        v_t[:] = self.beta2 * v_t + (1. - self.beta2) * grad * grad
        vMax_t[:] = maximum(v_t, vMax_t)
        
        # Rectify momentum
        mu = self.mu
        m_rec = lr *(1.0 -mu) *m_t + lr *mu * grad
        
        weight[:] -= m_rec / sqrt(vMax_t)