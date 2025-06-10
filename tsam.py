import torch

class TSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(TSAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step2(self, epsilon='gaussian', r=1, scaling=1, zero_grad=False):
        #norm_weight = 0
        norm_noise = 0
        count = 0
        noise = dict()
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["old_p"] = p.data.clone()
                if epsilon == 'gaussian':
                    noise[p] = scaling * torch.randn_like(p.data)
                elif epsilon == 'uniform':
                    noise[p] = torch.randn_like(p.data)
                norm_noise += noise[p].norm(2).item() ** 2
        
        norm_noise = norm_noise ** 0.5
        clip_bound = min(norm_noise, r)
        #print('norm noise:', norm_noise)
        if epsilon == 'uniform':
            scaling = torch.pow(torch.rand(1), 1.0 / 1e8).to('cuda')
        for group in self.param_groups:
            for p in group["params"]:
                if epsilon == 'gaussian':
                    noise[p] = noise[p] * clip_bound / norm_noise
                elif epsilon == 'uniform':
                    noise[p] = r * scaling * noise[p] / norm_noise
                p.add_(noise[p])
        
        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def first_step3(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        
        if zero_grad: self.zero_grad()

    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def tilted_aggregation(self, model, losses, grads, tilt=0.1):
        # perform tilted aggregation
        final_grads = dict()
        losses = torch.FloatTensor(losses)
        max_loss = torch.max(losses)
        
        # tilted weights of each gradient
        weights = torch.exp(tilt * (losses - max_loss)) / torch.sum(torch.exp(tilt * (losses - max_loss)))
        print('losses:', losses, 'weights:', weights)
        

        tmp_grad = dict()
        for p_name, p in model.named_parameters():
            tmp_grad[p_name] = torch.zeros_like(p)
        
        for p_name, p in model.named_parameters():
            for i, grad in enumerate(grads):
                # grad[p_name] = grad[p_name] * weights[i]
                tmp_grad[p_name].add_(grad[p_name] * weights[i])
            p.grad = tmp_grad[p_name]
        
        self.base_optimizer.step()
    
    @torch.no_grad()
    def simple_aggregation(self, model, grads):
        # perform simple aggregation
        final_grads = dict()
        n = len(grads)
        tmp_grad = dict()
        for p_name, p in model.named_parameters():
            tmp_grad[p_name] = torch.zeros_like(p)
        
        for p_name, p in model.named_parameters():
            for i, grad in enumerate(grads):
                tmp_grad[p_name].add_(grad[p_name] / n)
            p.grad = tmp_grad[p_name]
        
        self.base_optimizer.step()
    
    @torch.no_grad()
    def set_old_theta(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        
        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of multiple devices
        norm = torch.norm(
            torch.stack([
                (torch.abs(p) if group["adaptive"] else 1.0) * p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]), p=2
        )
        return norm
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
