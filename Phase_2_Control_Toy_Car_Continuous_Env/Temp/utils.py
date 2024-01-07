import torch

class shared_optim(torch.optim.Adam):
    def __init__(self, params, lr=5e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(shared_optim, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(param.data)
                state['exp_avg_sq'] = torch.zeros_like(param.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()