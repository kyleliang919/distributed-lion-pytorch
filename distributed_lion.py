from typing import Tuple, Optional, Callable
import torch
from torch.optim.optimizer import Optimizer

# functions

def exists(val):
    return val is not None

# update functions

def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay

    p.data.mul_(1 - lr * wd)

    # weight update

    update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1 - beta1).sign_()
    p.add_(update, alpha = -lr)

    # decay the momentum running average coefficient

    exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)

# class

class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay
        )

        super().__init__(params, defaults)

        self.update_fn = update_fn

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    wd,
                    beta1,
                    beta2
                )

        return loss


def update_fn_distributed(p, grad, state, lr, wd, beta1, beta2):
    state["simulated_counter"] += 1

    exp_avg = state["exp_avg"]
    simulated_accum_steps = len(exp_avg)
    # stepweight decay
    if state["simulated_counter"] >= simulated_accum_steps:
        p.data.mul_(1 - lr * wd)

    # weight update
    curr_exp_avg = exp_avg[state["simulated_counter"] - 1] 
    if not curr_exp_avg.is_cuda:
        curr_exp_avg_to_cuda = curr_exp_avg.cuda()

    update = curr_exp_avg_to_cuda.clone().mul_(beta1).add(grad, alpha = 1 - beta1).sign_()
    state["update_buffer"] += update / simulated_accum_steps

    curr_exp_avg_to_cuda.mul_(beta2).add_(grad, alpha = 1 - beta2)
    exp_avg[state["simulated_counter"] - 1] = curr_exp_avg_to_cuda if exp_avg[state["simulated_counter"] - 1].is_cuda else curr_exp_avg_to_cuda.cpu()

    if state["simulated_counter"] >= simulated_accum_steps:
        p.add_(state["update_buffer"], alpha = -lr)
        state["update_buffer"] = 0
        state["simulated_counter"] = 0
    
    
    

class LionDistributed(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        simulated_accum_steps = 8,
        cpu_offloading = True
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay
        )

        super().__init__(params, defaults)

        self.update_fn = update_fn_distributed
        self.simulated_accum_steps = simulated_accum_steps
        self.cpu_offloading = cpu_offloading

        

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None
    ):
        
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(filter(lambda p: exists(p.grad), group['params'])):

                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]
                    
                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    if not self.cpu_offloading:
                        state['exp_avg'] = [torch.zeros_like(p.data) for _ in range(self.simulated_accum_steps)]
                    else:
                        state['exp_avg'] = [torch.zeros_like(p.data).cpu() for _ in range(self.simulated_accum_steps)]
                    state["simulated_counter"] = 0
                    state["update_buffer"] = 0

                self.update_fn(
                    p,
                    grad,
                    state,
                    lr,
                    wd,
                    beta1,
                    beta2
                )
        return loss

class DGC(Optimizer):
    """
    u_t+1 = m * (u_t + g_t)
    v_t+1 = v_t + u_t + g_t
    thr = s% of |v_t+1|
    mask = v_t+1 > thr
    update = v_t+1 * mask
    v_t+1 = v_t+1 * (1 - mask)
    u_t+1 = u_t+1 * (1 - mask)
    """
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta: float = 0.9,
        weight_decay: float = 0.0,
    ):
        assert lr > 0.0
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group["params"]):
                grad, lr, wd, beta, state = (
                    p.grad,
                    group["lr"],
                    group["weight_decay"],
                    group["beta"],
                    self.state[p],
                )
                if len(state) == 0:
                    state["u"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                p.data.mul_(1 - lr * wd)
                u = state["u"]
                v = state["v"]
                u.add_(grad + u, alpha=beta)
                v.add_(grad + u)
                thr = torch.min(torch.topk(v.abs().view(-1), int(p.numel() * 0.25), largest=False)[0]) # s% of |v_t+1|, s = 0.25
                mask = v.abs() > thr
                update = v * mask
                v.mul_(torch.logical_not(mask))
                u.mul_(torch.logical_not(mask))
                p.add_(update, alpha=-lr)
        return loss
    
class DGCDistributed(Optimizer):
    """
    u_t+1 = m * (u_t + g_t)
    v_t+1 = v_t + u_t + g_t
    thr = s% of |v_t+1|
    mask = v_t+1 > thr
    update = v_t+1 * mask
    v_t+1 = v_t+1 * (1 - mask)
    u_t+1 = u_t+1 * (1 - mask)
    """
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta: float = 0.9,
        weight_decay: float = 0.0,
        simulated_accum_steps = 8,
        cpu_offloading = True
    ):
        assert lr > 0.0
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.simulated_accum_steps = simulated_accum_steps
        self.cpu_offloading = cpu_offloading
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(filter(lambda p: exists(p.grad), group["params"])):
                grad, lr, wd, beta, state = (
                    p.grad,
                    group["lr"],
                    group["weight_decay"],
                    group["beta"],
                    self.state[p],
                )
                if len(state) == 0:
                    if not self.cpu_offloading:
                        state['u'] = [torch.zeros_like(p.data) for _ in range(self.simulated_accum_steps)]
                        state['v'] = [torch.zeros_like(p.data) for _ in range(self.simulated_accum_steps)]
                    else:
                        state['u'] = [torch.zeros_like(p.data).cpu() for _ in range(self.simulated_accum_steps)]
                        state['v'] = [torch.zeros_like(p.data).cpu() for _ in range(self.simulated_accum_steps)]
                    state["simulated_counter"] = 0
                    state["update_buffer"] = 0

                state["simulated_counter"] += 1
                if state["simulated_counter"] >= self.simulated_accum_steps:
                    p.data.mul_(1 - lr * wd)

                u = state["u"][state["simulated_counter"] - 1]
                v = state["v"][state["simulated_counter"] - 1]

                if self.cpu_offloading:
                    u = u.cuda()
                    v = v.cuda()
                        
                u.add_(grad + u, alpha=beta)
                v.add_(grad + u)
                thr = torch.min(torch.topk(v.abs().view(-1), int(p.numel() * 0.25), largest=False)[0]) # s% of |v_t+1|, s = 0.25
                mask = v.abs() > thr
                update = v * mask
                state["update_buffer"] += update / self.simulated_accum_steps

                v.mul_(torch.logical_not(mask))
                u.mul_(torch.logical_not(mask))

                if self.cpu_offloading:
                    state["u"][state["simulated_counter"] - 1] = u.cpu()
                    state["v"][state["simulated_counter"] - 1] = v.cpu()
                else:
                    state["u"][state["simulated_counter"] - 1] = u
                    state["v"][state["simulated_counter"] - 1] = v



                if state["simulated_counter"] >= self.simulated_accum_steps:
                    p.add_(state["update_buffer"], alpha = -lr)
                    state["simulated_counter"] = 0
                    state["update_buffer"] = 0
        return loss