#Add the class to optimizer.py in MXNET to use the NAMSG optimizer
#Use set_obs_fac to set the observation distance eta
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
    eta : float, optional
        Oberservation distance
    """
    
    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.99, epsilon=1e-8, eta=0.9, **kwargs):
        super(Namsg, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.fEta = eta

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context, dtype=weight.dtype),  # momentum
                zeros(weight.shape, weight.context, dtype=weight.dtype),  # v
                self.epsilon * ones(weight.shape, weight.context, dtype=weight.dtype))  # vMax
    
    #set the observation distance eta    
    def set_obs_fac(self,fEta): 
        self.fEta=fEta

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
        fMu = self.fEta * (1.0/self.beta1 -1.0)
        m_rec = lr *(1.0 -fMu) *m_t + lr *fMu * grad
        
        weight[:] -= m_rec / sqrt(vMax_t)