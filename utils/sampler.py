import torch
from tqdm import tqdm

def euler_solver(model, x0, t_span, device, pbar=None):
    """Euler solver for ODE integration with progress bar"""
    x = x0
    dt = t_span[1] - t_span[0]
    
    for t_idx in range(len(t_span) - 1):
        t = t_span[t_idx] * torch.ones(x0.shape[0], device=device)
        dx = model(t, x)
        x = x + dx * dt
        if pbar is not None:
            pbar.update(1)
    
    return x

def heun_solver(model, x0, t_span, device, pbar=None):
    """Heun's solver (improved Euler) for ODE integration with progress bar"""
    x = x0
    dt = t_span[1] - t_span[0]
    
    for t_idx in range(len(t_span) - 1):
        t = t_span[t_idx] * torch.ones(x0.shape[0], device=device)
        t_next = t_span[t_idx + 1] * torch.ones(x0.shape[0], device=device)
        
        # First step: Euler
        dx1 = model(t, x)
        x_euler = x + dx1 * dt
        
        # Second step: Correction
        dx2 = model(t_next, x_euler)
        x = x + (dx1 + dx2) * dt / 2
        
        if pbar is not None:
            pbar.update(1)
    
    return x