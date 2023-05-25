import torch
from torch import nn
import numpy as np
import math
from scipy import integrate



# Beta functions for the noise scheduler from original papers

def linear_orig(t, max_, min_, T):
    return ((max_-min_)/T)*t + min_

def cosine_orig(t, s, T):
    if not type(t) is torch.Tensor:
        return 1-((math.cos((math.pi*(t+T*s))/(2*T*(s+1))))**2)*(((1/math.cos((math.pi*(t-1+T*s))/(2*T*(s+1)))))**2)
    return 1-((torch.cos((torch.pi*(t+T*s))/(2*T*(s+1))))**2)*(((1/torch.cos((torch.pi*(t-1+T*s))/(2*T*(s+1)))))**2)


# noise schedules (cumprod function): https://arxiv.org/abs/2301.10972

def linear_new(t, T, clip_min = 1e-9):
    if not type(t) is torch.Tensor:
        return max(1-t/T, clip_min)
    return (1 - t/T).clamp(min = clip_min)

def cosine(t, T, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    if not type(t) is torch.Tensor:
        v_start = math.cos(start * math.pi / 2) ** power
        v_end = math.cos(end * math.pi / 2) ** power
        output = math.cos((t/T * (end - start) + start) * math.pi / 2) ** power
        output = (v_end - output) / (v_end - v_start)
        return max(output, clip_min)
    start = torch.tensor(start, dtype=t.dtype, device=t.device)
    end = torch.tensor(end, dtype=t.dtype, device=t.device)
    tau = torch.tensor(tau, dtype=t.dtype, device=t.device)
    v_start = torch.cos(start * torch.pi / 2) ** power
    v_end = torch.cos(end * torch.pi / 2) ** power
    output = torch.cos((t/T * (end - start) + start) * torch.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid(t, T, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t/T * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)


# Converts cumprod functions to individual betas

def cumprod_to_beta(t, cumprods):
    return 1-(
        cumprods[t+1] / cumprods[t]
    )




class Scheduler():
    def __init__(self,
                 num_steps=1000):
        
        self.num_steps = num_steps
        self.timesteps = torch.tensor(np.linspace(0, num_steps, num_steps+1), dtype=torch.int).cpu()
    
    def get_sampling_timesteps(self, batch, device):
        return tuple(torch.tensor([[self.timesteps[i]], [self.timesteps[i-1]]], dtype=torch.int32, device=device).repeat(batch, 1) for i in range(self.num_steps, 0, -1))
    
    def sample_random(self, batch, device):
        return self.timesteps[np.random.randint(0, self.num_steps, batch)].to(device)
    
    def get_cumprods(self, times, device):
        pass
    
    def get_lambdas(self, times, device):
        pass
    
    def get_lambdas_inv(self, times, device):
        pass
    
    def get_ps(self, times, device):
        pass




class LinearScheduler(Scheduler):
    def __init__(self,
                 num_steps=1000,
                 min_value=1e-4,
                 max_value=0.02,
                 original = False):
        super().__init__(num_steps)
        
        # Variance scheduler beta values
        if original:
            self.betas = torch.tensor(linear_orig(self.timesteps, max_value, min_value, num_steps), dtype=torch.float32).cpu()
            self.alphas = 1-self.betas
            self.alphas_cumprod = torch.tensor(np.cumprod(self.alphas), dtype=torch.float32).cpu()
        else:
            # self.alphas_cumprod = torch.tensor(linear(self.timesteps, min_value), dtype=torch.float32).cpu()
            # max_value = 1
            raise NotImplementedError("New linear scheduler not implemented yet")
        
        # Calculate implicit integral from 0 to t
        def integral_calc(t):
            return ((max_value-min_value)/(2*num_steps))*t**2 + min_value*t
        self.integrals = torch.tensor([integral_calc(t) for t in self.timesteps], dtype=torch.float32).cpu()
        
        # Calculate the p and lambda values
        self.p = torch.e**(-0.5*self.integrals)
        self.lambdas = (1-torch.exp(-self.integrals))
        self.lambdas_inv = 1/self.lambdas
        
        
        
    def get_cumprods(self, times, device):
        return self.alphas_cumprod.to(device)[times]
            
    def get_lambdas(self, times, device):
        return self.lambdas.to(device)[times]
    
    def get_lambdas_inv(self, times, device):
        return self.lambdas_inv.to(device)[times]
    
    def get_ps(self, times, device):
        return self.p.to(device)[times]
    
    
    
    
    
    
class CosineScheduler(Scheduler):
    def __init__(self,
                 num_steps=1000,
                 s=0.008,
                 original=True):
        super().__init__(num_steps)
        
        # Variance scheduler beta values
        if original:
            self.betas = torch.tensor(cosine_orig(self.timesteps, s, num_steps), dtype=torch.float32).cpu()
            self.alphas = 1-self.betas
            self.alphas_cumprod = torch.tensor(np.cumprod(self.alphas), dtype=torch.float32).cpu()
        else:
            self.alphas_cumprod = torch.tensor(cosine(self.timesteps, num_steps), dtype=torch.float32).cpu()
            raise NotImplementedError("New cosine scheduler not implemented yet")
        
        # Calculate implicit integral from 0 to t
        # Since the cosine scheduler has a really complicated
        # function, it's easier to just estimate the integral
        def integral_calc(t):
            return integrate.quad(cosine_orig, 0, t, args=(s, num_steps))[0]
        self.integrals = torch.tensor([integral_calc(t) for t in self.timesteps], dtype=torch.float32).cpu()
        
        # Calculate the p and lambda values
        self.p = torch.e**(-0.5*self.integrals)
        self.lambdas = (1-torch.exp(-self.integrals))
        self.lambdas_inv = 1/self.lambdas
        
        
        
    def get_cumprods(self, times, device):
        return self.alphas_cumprod.to(device)[times]
            
    def get_lambdas(self, times, device):
        return self.lambdas.to(device)[times]
    
    def get_lambdas_inv(self, times, device):
        return self.lambdas_inv.to(device)[times]
    
    def get_ps(self, times, device):
        return self.p.to(device)[times]
    
    
    
    
    
    
class SigmoidScheduler(Scheduler):
    def __init__(self,
                 num_steps=1000,
                 start=-3,
                 end=3,
                 tau=1,
                 clamp_min=1e-9):
        super().__init__(num_steps)
        
        # Variance scheduler beta values
        self.alphas_cumprod = torch.tensor(sigmoid(self.timesteps, num_steps, start, end, tau, clamp_min), dtype=torch.float32).cpu()
        def a(t, T, start, end, tau, clamp_min):
            start = torch.tensor(start)
            end = torch.tensor(end)
            tau = torch.tensor(tau)
            v_end = torch.tensor(end / tau).sigmoid()
            num = v_end - ((t*(end-start) + start) / tau).sigmoid()
            den = v_end - (((t-1)*(end-start) + start) / tau).sigmoid()
            return 1- (num/den)
        a(self.timesteps, num_steps, start, end, tau, clamp_min)
        
        # Calculate implicit integral from 0 to t
        # Since the cosine scheduler has a really complicated
        # function, it's easier to just estimate the integral
        def integral_calc(t):
            return integrate.quad(sigmoid, 0, t, args=(num_steps, start, end, tau, clamp_min))[0]
        self.integrals = torch.tensor([integral_calc(t) for t in self.timesteps], dtype=torch.float32).cpu()
        
        # Calculate the p and lambda values
        self.p = torch.e**(-0.5*self.integrals)
        self.lambdas = (1-torch.exp(-self.integrals))
        self.lambdas_inv = 1/self.lambdas
        
        
        
    def get_cumprods(self, times, device):
        return self.alphas_cumprod.to(device)[times]
            
    def get_lambdas(self, times, device):
        return self.lambdas.to(device)[times]
    
    def get_lambdas_inv(self, times, device):
        return self.lambdas_inv.to(device)[times]
    
    def get_ps(self, times, device):
        return self.p.to(device)[times]