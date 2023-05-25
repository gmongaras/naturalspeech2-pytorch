import torch
from torch import nn
import numpy as np
import math
from scipy import integrate



# Beta functions for the noise scheduler
def linear(max_, min_, T, t):
    return ((max_-min_)/T)*t + min_

def cosine(s, T, t):
    return 1-((math.cos((math.pi*(t+T*s))/(2*T*(s+1))))**2)*(((1/math.cos((math.pi*(t-1+T*s))/(2*T*(s+1)))))**2)




class LinearScheduler():
    def __init__(self,
                 device,
                 num_steps=1000,
                 min_value=1e-4,
                 max_value=0.02):
        super(LinearScheduler, self).__init__()
        
        self.num_steps = num_steps
        self.timesteps = np.linspace(0, num_steps, num_steps+1).astype(int)
        
        # Variance scheduler beta values
        self.betas = linear(max_value, min_value, num_steps, self.timesteps)
        self.alphas = 1-self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
        # Calculate implicit integral from 0 to t
        def integral_calc(t):
            return ((max_value-min_value)/(2*num_steps))*t**2 + min_value*t
        self.integrals = np.array([integral_calc(t) for t in self.timesteps])
        
        # Calculate the p and lambda values
        self.p = np.e**(-0.5*self.integrals)
        self.lambdas = (1-np.exp(-self.integrals))
        self.lambdas_inv = 1/self.lambdas
        
        
        
    def get_sampling_timesteps(self, batch, device):
        return tuple(torch.tensor([[self.timesteps[i]], [self.timesteps[i-1]]], dtype=torch.int32, device=device).repeat(batch, 1) for i in range(self.num_steps, 0, -1))
    
    
    def get_alphas_sigmas(self, time, batch, device):
        return torch.sqrt(torch.tensor(self.alphas_cumprod, dtype=torch.float32, device=device)[time]).repeat(batch),\
            torch.sqrt(1-torch.tensor(self.alphas_cumprod, dtype=torch.float32, device=device)[time]).repeat(batch)
            
            
    def get_alphas_betas_cumprod(self, time, batch, device):
        return torch.tensor(self.alphas, dtype=torch.float32, device=device)[time].repeat(batch),\
            torch.tensor(self.betas, dtype=torch.float32,  device=device)[time].repeat(batch),\
            torch.tensor(self.alphas_cumprod, dtype=torch.float32,device=device)[time].repeat(batch)
            
    def get_lambdas(self, times, device):
        return torch.tensor(self.lambdas, dtype=torch.float32, device=device)[times]
    
    def get_lambdas_inv(self, times, device):
        return torch.tensor(self.lambdas_inv, dtype=torch.float32, device=device)[times]
    
    def get_ps(self, times, device):
        return torch.tensor(self.p, dtype=torch.float32, device=device)[times]
            
    def sample_random(self, batch, device):
        return torch.tensor(self.timesteps[np.random.randint(0, self.num_steps, batch)], dtype=torch.int, device=device)
        
        
    
    
    def get_steps(self,
                  sample_steps=None):

        if sample_steps is None:
            sample_steps = self.num_steps
            
        return self.scheduler[::(self.num_steps // sample_steps)]
    
    
    
    
    
    
    
    
    
# noise schedules

def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)