import torch

class SSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                grad = p.grad.data

                # Calculate Hessian-vector product and adjust gradient
                # hvp = self._approx_hessian_vector_product(closure, p, grad)
                hvp = torch.autograd.functional.hvp(closure, [param], [grad])[1]
                adjusted_grad = grad + 0.5 * hvp
                adjusted_grad.mul_(scale)

                # Update parameter
                p.add_(adjusted_grad)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Restore to the old parameters
                p.data = self.state[p]["old_p"]

        # Perform five optimization steps
        for _ in range(5):
            self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    # def _approx_hessian_vector_product(self, closure, param, grad):
    #     # This function computes the Hessian-vector product using a finite difference approximation
    #     # For more accurate calculation, consider using torch.autograd.functional.hvp()
    #     orig_data = param.data.clone()
    #     perturbed_data = orig_data + 1e-2 * grad
    #     param.data = perturbed_data
    #     loss = closure()
    #     perturbed_grad = torch.autograd.grad(loss, param)[0]

    #     param.data = orig_data  # restore original data
    #     return (perturbed_grad - grad) / 1e-2

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

# Usage
model = torch.nn.Linear(10, 2)
optimizer = SSAM(model.parameters(), base_optimizer=torch.optim.SGD, lr=0.01, rho=0.1)

def closure():
    optimizer.zero_grad()
    output = model(torch.randn(10))
    loss = output.sum()
    loss.backward()
    return loss

for _ in range(10):
    optimizer.step(closure)



### GPT Version:

# import torch

# class SSAM(torch.optim.Optimizer):
#     def __init__(self, params, base_optimizer, rho=0.05, lr=0.1, **kwargs):
#         assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
#         self.base_optimizer = base_optimizer(params, lr=lr, **kwargs)
#         self.defaults = dict(rho=rho, lr=lr, **kwargs)
#         super(SSAM, self).__init__(params, self.defaults)

#     @torch.no_grad()
#     def step(self, closure):
#         assert closure is not None, "Sharpness Aware Minimization requires a closure that returns the loss"

#         # First gradient ascent step
#         loss = closure()
#         self.first_step()

#         # Recalculate the loss and do the second step multiple times
#         loss = closure()
#         self.second_step()

#         return loss

#     def first_step(self, zero_grad=False):
#         grad_norm = self._grad_norm()
#         for group in self.param_groups:
#             scale = group["rho"] / (grad_norm + 1e-12)
            
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 self.state[p]["old_p"] = p.data.clone()
#                 grad = p.grad.data

#                 # Calculate Hessian-vector product and adjust gradient
#                 hvp = self._approx_hessian_vector_product(closure, p, grad)
#                 adjusted_grad = grad + 0.5 * hvp
#                 adjusted_grad.mul_(scale)

#                 # Update parameter
#                 p.add_(adjusted_grad)

#         if zero_grad:
#             self.zero_grad()

#     def second_step(self, zero_grad=False):
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 # Restore to the old parameters
#                 p.data = self.state[p]["old_p"]

#         # Perform five optimization steps
#         for _ in range(5):
#             self.base_optimizer.step()

#         if zero_grad:
#             self.zero_grad()

#     def _grad_norm(self):
#         shared_device = self.param_groups[0]["params"][0].device
#         norm = torch.norm(
#             torch.stack([
#                 p.grad.norm(2).to(shared_device)
#                 for group in self.param_groups for p in group["params"] if p.grad is not None
#             ]),
#             p=2
#         )
#         return norm

#     def _approx_hessian_vector_product(self, closure, param, grad):
#         # This function computes the Hessian-vector product using a finite difference approximation
#         # For more accurate calculation, consider using torch.autograd.functional.hvp()
#         orig_data = param.data.clone()
#         perturbed_data = orig_data + 1e-2 * grad
#         param.data = perturbed_data
#         loss = closure()
#         perturbed_grad = torch.autograd.grad(loss, param)[0]

#         param.data = orig_data  # restore original data
#         return (perturbed_grad - grad) / 1e-2

# # Usage
# model = torch.nn.Linear(10, 2)
# optimizer = SSAM(model.parameters(), base_optimizer=torch.optim.SGD, lr=0.01, rho=0.1)

# def closure():
#     optimizer.zero_grad()
#     output = model(torch.randn(10))
#     loss = output.sum()
#     loss.backward()
#     return loss

# for _ in range(10):
#     optimizer.step(closure)
