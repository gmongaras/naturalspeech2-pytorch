import torch
from torch import nn
import numpy as np
import math
from scipy import integrate
import multiprocessing



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

def cosine_new(t, T, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    t = torch.tensor(t, dtype=torch.float32) if not type(t) is torch.Tensor else t.to(torch.float32)
    start = torch.tensor(start, dtype=torch.float32, device=t.device) if not type(start) is torch.Tensor else start
    end = torch.tensor(end, dtype=torch.float32, device=t.device) if not type(end) is torch.Tensor else end
    tau = torch.tensor(tau, dtype=torch.float32, device=t.device) if not type(tau) is torch.Tensor else tau
    power = 2 * tau
    v_start = torch.cos(start * torch.pi / 2) ** power
    v_end = torch.cos(end * torch.pi / 2) ** power
    output = torch.cos((t/T * (end - start) + start) * torch.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid(t, T, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    t = torch.tensor(t, dtype=torch.float32) if not type(t) is torch.Tensor else t
    start = torch.tensor(start, dtype=t.dtype, device=t.device) if not type(start) is torch.Tensor else start
    end = torch.tensor(end, dtype=t.dtype, device=t.device) if not type(end) is torch.Tensor else end
    tau = torch.tensor(tau, dtype=t.dtype, device=t.device) if not type(tau) is torch.Tensor else tau
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t/T * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)


# Converts cumprod functions to individual betas

def cumprod_to_beta(t, funct, args):
    return 1-(
        funct(t+1, *args) / funct(t, *args)
    )
    
# Multiprocess an integral calculation

def worker(t, function, out_dict):
    for subset in t:
        out_dict[subset] = function(subset)
def multiprocess_integral_calc(function, timesteps, num_workers):
    manager = multiprocessing.Manager()
    out_arr = manager.list([torch.tensor(0, dtype=torch.float32, device=timesteps.device) for t in timesteps])
    
    # manager.start()
    jobs = []
    num_timesteps = len(timesteps)
    prop = num_timesteps/num_workers                    
    times = [[j for j in range(math.ceil(prop*i), math.floor(prop*(i+1)))] for i in range(0, num_workers)]
    for t in times:
        p = multiprocessing.Process(target=worker, args=(t, function, out_arr))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
        
    return [i for i in out_arr]
    




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
            self.alphas_cumprod = torch.tensor(linear_new(self.timesteps, min_value), dtype=torch.float32).cpu()
            max_value = 1
            # raise NotImplementedError("New linear scheduler not implemented yet")
        
        # Calculate implicit integral from 0 to t
        def integral_calc(t):
            if original:
                # Calculating the integral of the function, we get ((min-max)/2T) * t^2 + min*t
                return ((max_value-min_value)/(2*num_steps))*t**2 + min_value*t
            else:
                # The function 1-t/T is the cumulative product. We can spcifically model the
                # beta function as the difference between the next step and current step
                # by beta = 1-(cumprod[t+1]/cumprod[t]). This is the integral of the beta function
                # which is ln|(T-t)/T)| + t.
                return torch.log(torch.abs((num_steps-t)/num_steps)) + t
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
    
    
    
    
    
    
@torch.no_grad()
class CosineScheduler(Scheduler):
    def __init__(self,
                 num_steps=1000,
                 s=0.008,
                 start=0,
                 end=1,
                 tau=1,
                 clamp_min=1e-9,
                 original=False):
        super().__init__(num_steps)
        
        # Variance scheduler beta values
        if original:
            self.betas = torch.tensor(cosine_orig(self.timesteps, s, num_steps), dtype=torch.float32).cpu()
            self.alphas = 1-self.betas
            self.alphas_cumprod = torch.tensor(np.cumprod(self.alphas), dtype=torch.float32).cpu()
        else:
            self.alphas_cumprod = torch.tensor(cosine_new(self.timesteps, num_steps, start, end, tau, clamp_min), dtype=torch.float32).cpu()
        
        # Calculate implicit integral from 0 to t
        # Since the cosine scheduler has a really complicated
        # function, it's easier to just estimate the integral.
        # Note that since the new cosine scheduler is in terms
        # of cumulative products, we must model it as 1-(cumprod[t+1]/cumprod[t])
        # and take the integral of that function instead.
        def integral_calc(t):
            return integrate.quad(cosine_orig, 0, t, args=(s, num_steps))[0] if original \
                else integrate.quad(cumprod_to_beta, 0, t, args=(cosine_new, (num_steps, start, end, tau, clamp_min)))[0]
        # self.integrals = torch.tensor([integral_calc(t) for t in self.timesteps], dtype=torch.float32).cpu()
        self.integrals = torch.tensor(multiprocess_integral_calc(integral_calc, self.timesteps, 20), dtype=torch.float32).cpu()
        
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
            return integrate.quad(cumprod_to_beta, 0, t, args=(sigmoid, (num_steps, start, end, tau, clamp_min)))[0]
        self.integrals = torch.tensor(multiprocess_integral_calc(integral_calc, self.timesteps, 20), dtype=torch.float32).cpu()
        # def integral_calc(t):
        #     return integrate.quad(sigmoid, 0, t, args=(num_steps, start, end, tau, clamp_min))[0]
        # self.integrals = torch.tensor([integral_calc(t) for t in self.timesteps], dtype=torch.float32).cpu()
        
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