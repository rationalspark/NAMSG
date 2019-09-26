#Add the class to optimizer.py in MXNET to use the NAMSG optimizer

#Recommended hyper-parameters: beta1=0.999, beta2=0.99, epsilon=1e-8,
#Use mu=0.1 to maximize training speed, and mu=0.2 to improve generalization.
#learning_rate is obtained through grid search.

#The observation boost hyper-parameter (OBSB) policy
#Perform grid search for a small proportion of epoches to obtain the initial step size alpha0,
#with beta1=0.999, beta2=0.99, and mu=mu0=0.05. 
#In training, start with step size alpha0 and observation factor mu0. 
#Compute the average convergence rate at the interval of several epoches.
#In experiments to optimize training speed, apply boost when the convergence rate is halved.
#In experiments to optimize generalization, apply boost when the train loss flattens, 
#or when the validation accuracy flattens if a validation set is available.
#To apply boost, set mu=2*mu, and scale the step size according to the ratio of 
#tau=alpha*lambda to minimize the gain factor. If mu0=0.05, alpha=alpha0*0.247.
#OBSB is different from vanilla learning_rate decay, since a small mu enalbes large initial
#step size. After applying the observation boost, the step size is still relatively large.
#Consequently, we can still apply learning_rate decay to achieve acceleration.

#Use set_obs_fac_mu to set the observation factor mu.

@register
class Namsg(Optimizer):
    """
    The NAMSG optimizer
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
    
    def __init__(self, learning_rate=0.002, beta1=0.999, beta2=0.99, epsilon=1e-8, mu=0.1, **kwargs):
        super(Namsg, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.fMu = mu

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context, dtype=weight.dtype),  # momentum
                zeros(weight.shape, weight.context, dtype=weight.dtype),  # v
                self.epsilon * ones(weight.shape, weight.context, dtype=weight.dtype))  # vMax
    
    #set the observation factor 
    def set_obs_fac_mu(self,fMu): 
        self.fMu=fMu

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
        fMu = self.fMu
        m_rec = lr *(1.0 -fMu) *m_t + lr *fMu * grad
        
        weight[:] -= m_rec / sqrt(vMax_t)
        
       