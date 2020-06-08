import math
import torch
from .optimizer import Optimizer

class Arsg(Optimizer):
    '''Implements RSG and ARSG algorithm.

    It has been proposed in `An Adaptive Remote Stochastic Gradient Method for Training Neural Networks`.
    The parameters and states are vectorized for acceleration. As a side effect, muti-groups of hyper-parameters are difficult to implement in this maner. Hence it is not supported yet.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3). It should be selected by grid search to optimize performance
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.999, 0.99))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8). Set to 1e-3 in fine mode to improve generalization
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        mu (float, optional): observation factor (default: 0.1)
        adaptive (boolean, optional): whether to use an adaptive preconditioner.
            Disable the term to use the remote stochastic gradient method (RSG).
            When mu=0, RSG is equal to SGD with lr = lr/(1-beta)
        amsgrad (boolean, optional): whether to use the adaptive preconditioner modified from 
            AMSGRAD `On the Convergence of Adam and Beyond`_(default: True). 
            Set to False to use the preconditioner similar to ADAM
        nLA (integer, optional): lookahead inner loop number, minus to disable (default: -1)
            It is not recommended to use lookahead optimizer alongwith ARSG with the default hyper-parameters.
        alphaLA: update ratio for slow weights in the lookahead mode (default: 0.5)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    .. _Lookahead Optimizer: k steps forward, 1 step back
    .. _An Adaptive Remote Stochastic Gradient Method for Training Neural Networks
        https://arxiv.org/abs/1905.01422    
    '''

    def __init__(self, params, lr=1e-3, betas=(0.999, 0.99), eps=1e-8,
                 weight_decay=0, mu=0.1, adaptive=True, amsgrad=True, nLA=-1, alphaLA=0.5):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= mu < 1.0:
            raise ValueError("Invalid mu: {}".format(mu))
        if type(nLA)!=int:
            raise ValueError("Invalid nLA: {}".format(nLA))
        if nLA > 1 and not 0.0 <= alphaLA <= 1.0:
            raise ValueError("Invalid alphaLA: {}".format(alphaLA))
        if not adaptive and amsgrad:
            amsgrad = False
            print('AMSGrad is not valid in non-adaptive mode, disabled.')

        #Default initialization            
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, mu=mu, adaptive=adaptive, amsgrad=amsgrad, nLA=nLA, alphaLA=alphaLA)
        super(Arsg, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        super(Arsg, self).__setstate__(state)

    def step(self, closure=None, accumulate_grad=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        #Get loss
        loss = None
        if closure is not None:
            loss = closure()
        # Get group and state
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                break
            break
        # Get hyperparameters
        amsgrad = group['amsgrad']
        adaptive = group['adaptive']
        lr = group['lr']
        beta1, beta2 = group['betas']
        mu = group['mu']
        weight_decay = group['weight_decay']
        eps = group['eps']  
        nLA = group['nLA']
        alphaLA = group['alphaLA']
        # State initialization
        if not hasattr(self,'nLenData'):
            assert (len(self.param_groups)==1), 'Muti-group of hyper-parameters are not supported yet'
            self.nLenData, self.listParaLen, self.device = self.get_len_and_device()           
        if len(state) == 0:
            state['step'] = 1
            state['nSubBatch'] = 0
            state['data'] = torch.zeros(self.nLenData, device=self.device)
            self.para_to_vec(state['data'])
            state['grad'] = torch.zeros_like(state['data'])
            state['exp_avg'] = torch.zeros_like(state['data'])
            if adaptive:
                state['exp_avg_sq'] = torch.zeros_like(state['data'])
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(state['data']).add_(eps/(1-beta2))
            if nLA > 1:
                state['data_slow'] = state['data'].clone()
                #state['exp_avg_slow'] = torch.zeros_like(state['data'])
        #Get states
        data = state['data']
        grad = state['grad']
        exp_avg = state['exp_avg']
        if adaptive:
            exp_avg_sq = state['exp_avg_sq']
            if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
        #Get gradient
        if state['nSubBatch'] == 0:
            self.grad_to_vec(grad)
        else:
            gradSubBatch = torch.zeros_like(data)
            self.grad_to_vec(gradSubBatch)
            grad.add_(gradSubBatch)
        state['nSubBatch'] += 1    
        #Return in gradient accumulating steps
        if accumulate_grad:
            return loss
        if state['nSubBatch'] >1:
            grad /= state['nSubBatch']
        state['nSubBatch'] = 0
        #Updation
        if weight_decay != 0:
            grad.add_(weight_decay, data)
        if adaptive:
            if amsgrad:
                #RSG & AMSGRAD
                exp_avg.mul_(beta1).add_(grad)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad)
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = max_exp_avg_sq.sqrt()
                amp_step_numerator = (1-beta1) * (1-mu)
                step_size1 = lr * amp_step_numerator/ (1-beta2)**0.5
                data.addcdiv_(-step_size1, exp_avg + mu/amp_step_numerator * grad, denom)
            else:
                #RSG & ADAM
                exp_avg.mul_(beta1).add_(grad)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad)
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                amp_step_numerator = (1-beta1)* (1-mu) / bias_correction1
                amp_denom = ((1 - beta2)/bias_correction2)**0.5
                denom = exp_avg_sq.sqrt().add_(eps/amp_denom)
                step_size1 = lr * amp_step_numerator / amp_denom
                data.addcdiv_(-step_size1, exp_avg + mu/amp_step_numerator * grad, denom)
        else:
            #Original RSG
            exp_avg.mul_(beta1).add_(grad)
            step_size1 = lr * (1-beta1) * (1-mu) 
            data.add_(-step_size1, exp_avg + mu/(1-mu)/(1-beta1)*grad)
        #Lookahead
        if nLA > 1 and state['step'] % nLA == 0:
            state['data_slow'].add_((data-state['data_slow'])*alphaLA)
            #state['exp_avg_slow'].add_((exp_avg-state['exp_avg_slow'])*alphaLA)
            data.copy_(state['data_slow'])
            #exp_avg.copy_(state['exp_avg_slow'])
        self.vec_to_para(data)
        state['step'] += 1
        return loss

    def get_len_and_device(self):
        nLenData = 0
        listParaLen = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError('Arsg does not support sparse gradients, please consider SparseAdam instead')
                t1DataCur = p.data.flatten()
                nSizeCur = t1DataCur.size(0)
                nLenData += nSizeCur
                listParaLen.append(nSizeCur)
        return [nLenData, listParaLen, p.device]
        
    #Collect the trainable paramters to a pre-allocated 1d tensor. The length have to be consistant.
    def para_to_vec(self, t1Data):
        nPos = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                t1DataCur = p.data.flatten()
                nPosNext = nPos+t1DataCur.size(0)
                t1Data[nPos:nPosNext] = t1DataCur
                nPos = nPosNext

    #Transfer the data in a 1d tensor to the trainable paramters. The length have to be consistant.
    def vec_to_para(self, t1Data):
        nPos = 0
        nIndPara = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                nPosNext = nPos+self.listParaLen[nIndPara]
                t1DataCur = t1Data[nPos:nPosNext]
                p.data = t1DataCur.view(p.data.size())
                nPos = nPosNext
                nIndPara += 1

    #Collect gradient to a pre-allocated 1d tensor. The length have to be consistant.
    def grad_to_vec(self, t1Grad):
        nPos = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                t1GradCur = p.grad.data.flatten()
                nPosNext = nPos+t1GradCur.size(0)
                t1Grad[nPos:nPosNext] = t1GradCur
                nPos = nPosNext

    def restart(self):
        """Restart by setting vMax=v.
        """
        for group in self.param_groups:
            for p in group['params']:
                if not group['amsgrad']:
                    print('Restart is invalid in non_amsgrad mode, skipped')
                    return            
                if not p.requires_grad:
                    continue
                state = self.state[p]
                break
            break
        if len(state) == 0:
            print('Restart is invalid before state initialization, skipped')
            return            
        beta2 = group['betas'][1]
        eps = group['eps']
        state['max_exp_avg_sq'].copy_(state['exp_avg_sq'])
        max_exp_avg_sq = state['max_exp_avg_sq']
        torch.max(max_exp_avg_sq, torch.zeros_like(max_exp_avg_sq).add_(eps/(1-beta2)), out=max_exp_avg_sq)
 
    def set_hyper_para(self,lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, mu=0.1):
        """Set hyperparameters.
        """ 
        for group in self.param_groups:
            group['lr'] = lr
            group['betas'] = betas
            group['mu'] = mu
            group['weight_decay'] = weight_decay
            group['eps'] = eps
        # Modify state
        if group['amsgrad']:
            for group in self.param_groups:
                for p in group['params']:
                    if not p.requires_grad:
                        continue
                    state = self.state[p]
                    break
                break
            if len(state) == 0:
                print('Restart is invalid before state initialization, skipped')
                return
            beta2 = group['betas'][1]
            max_exp_avg_sq = state['max_exp_avg_sq']
            torch.max(max_exp_avg_sq, torch.zeros_like(max_exp_avg_sq).add_(eps/(1-beta2)), out=max_exp_avg_sq)
            

