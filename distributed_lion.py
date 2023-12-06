from typing import Tuple, Optional, Callable
import torch
from torch.optim.optimizer import Optimizer
import torch.distributed as dist

# functions

def exists(val):
    return val is not None

# util functions


def flatten_and_pad(tensor):
    # Flatten the tensor
    flattened_tensor = tensor.flatten()

    # Calculate the padding size to make the length a multiple of 8
    padding_size = (8 - (flattened_tensor.numel() % 8)) % 8

    # Pad the tensor with zeros
    padded_tensor = torch.nn.functional.pad(flattened_tensor, (0, padding_size))

    return padded_tensor, tensor.shape


def restore_flattened_tensor(padded_tensor, original_shape):
    # Remove padding
    restored_tensor = padded_tensor[:original_shape.numel()].reshape(original_shape)

    return restored_tensor

def majority_vote(boolean_tensors):
    # Stack boolean tensors along a new dimension
    stacked_tensor = torch.stack(boolean_tensors, dim=0)

    # Use the mode function to find the majority vote along the new dimension
    majority_voted_tensor, _ = torch.mode(stacked_tensor, dim=0)

    # Convert the result back to boolean
    majority_voted_tensor = majority_voted_tensor.bool()

    return majority_voted_tensor

# update functions

def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay

    p.data.mul_(1 - lr * wd)

    # weight update

    update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1 - beta1).sign_()
    p.add_(update, alpha = -lr)

    # decay the momentum running average coefficient

    exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)

def update_fn_distributed(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay
    
    p.data.mul_(1 - lr * wd)

    # weight update
    
    update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1 - beta1).sign_()

    # pad the flattened tensor to multiple of 8
    padded_update, udpate_shape = flatten_and_pad(update > 0)


    # reshape the tensor to prepare for comparession and define the indexes for bit shift
    bool_tensor = padded_update.view(-1, 8).unsqueeze(0)
    indexes = torch.arange(8).unsqueeze(0).unsqueeze(0).to(grad.device)
    uint8_tensor = (bool_tensor.byte() << indexes).sum(dim=-1)

    # all gather from all other workers
    all_updates = [torch.zeros_like(uint8_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(all_updates, uint8_tensor)

    # decode back into bool tensor
    bool_updates = []
    for each in all_updates:
        decoded = (each.unsqueeze(-1)>>indexes)%2==1
        flattened_decoded = decoded.squeeze().flatten()
        bool_updates.append(restore_flattened_tensor(flattened_decoded), udpate_shape)

    # majority vote on all the updates
    update = majority_vote(bool_updates) * 2 - 1
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
        self.update_fn = update_fn_distributed if dist.get_world_size() > 1 else update_fn

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
        cpu_offloading = True
    ):
        assert lr > 0.0
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)
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
                        #all_grads = [torch.zeros_like(p.grad) for _ in range(dist.get_world_size())]
                        #dist.all_gather(all_grads, p.grad.data)
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