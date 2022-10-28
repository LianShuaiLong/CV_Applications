from tqdm import tqdm

import torch
import math


import torch.nn.functional as F 

def linear_beta_schedule(timesteps):
    # 从scale*0.0001-scale*0.02之间等间隔取timesteps个数
    scale = 1000/timesteps
    beta_start = scale*0.0001
    beta_end = scale*0.02
    return torch.linspace(beta_start,beta_end,timesteps,dtype=torch.float64)

def cosine_beta_schedule(timesteps,s=0.008):
    steps = timesteps+1
    x = torch.linspace(0,timesteps,steps,dtype=torch.float64)
    alphas_cumprod = torch.cos(((x/timesteps)+s)/(1+s)*math.pi*0.5)**2
    alphas_cumprod = alphas_cumprod/alphas_cumprod[0]
    betas = 1-alphas_cumprod[1:]/alphas_cumprod[:-1]
    return torch.clip(betas,0,0.999)

class GaussianDiffusion:
    def __init__(self,timesteps=1000,beta_schedule = 'linear'):
        self.timesteps = timesteps
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'Invalid noise schedule:{beta_schedule}')
        self.betas = betas

        self.alphas = 1 - self.betas
        # Return the cumulative product of elements along a given axis.
        self.alphas_cumprod = torch.cumprod(self.alphas,axis=0)# -\overset\alpha_t

        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1],(1,0),value=1) # -\overset\alpha_{t-1}

        # caculations for diffusion q(x_t|x_{t-1})  扩散过程x_0----->x_T
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0-self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0-self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0/self.alphas_cumprod)
        self.sqrt_recipml_alphas_cumprod = torch.sqrt(1.0/self.alphas_cumprod-1)

        # calculations for posterior q(x_{t-1}|x_t,x_0) 真实的去噪过程 x_T----->x_0
        self.posterior_variance = (self.betas*(1.0-self.alphas_cumprod_prev)/(1-self.alphas_cumprod))
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        self.posterior_mean_cof_xt = torch.sqrt(self.alphas)*(1-self.alphas_cumprod_prev)/(1-self.alphas_cumprod)
        self.posterior_mean_cof_x0 = torch.sqrt(self.alphas_cumprod_prev)*self.betas/(1-self.alphas_cumprod)
         
    # get the params of given timestep t 
    def _extract(self,a,t,x_shape):
        batch_size = t.shape[0]  # batch_size为什么是t.shape[0]
        out = a.to(t.device).gather(0,t).float() 
        # gather按照索引取值，
        # torch.gather(input, dim, index, *, sparse_grad=False, out=None) → Tensor
        out = out.reshape(batch_size,*((1,)*(len(x_shape)-1)))
        return out 
    
    # forward diffusion,利用x_t~q(x_t|x_0)
    def q_sample(self,x_start,t,noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod,t,x_start.shape) #设计noise schedule,根据timestep=t gather处对应的sqrt_alpha_cumprod
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod,t,x_start.shape)

        return sqrt_alphas_cumprod_t*x_start+sqrt_one_minus_alphas_cumprod_t*noise
    
    # Get the mean and variance of q(x_t|x_0)
    def q_mean_variance(self,x_start,t):
        mean = self._extract(self.sqrt_alphas_cumprod,t,x_start.shape)*x_start
        variance = self._extract(1.0-self.alphas_cumprod,t,x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod,t,x_start.shape)
        return mean,variance,log_variance

    # compute the mean and variance of the diffusion posterior q(x_{t-1}|x_t,x_0)
    def q_posterior_mean_variance(self,x_start,x_t,t):
        posterior_mean = self._extract(self.posterior_mean_cof_xt,t,x_start.shape)*x_t+self._extract(self.posterior_mean_cof_x0,t,x_start.shape)*x_start
        posterior_variance = self._extract(self.posterior_variance,t,x_t.shape)
        posterior_log_variance_cliped = self._extract(self.posterior_log_variance_clipped,t,x_t.shape)
        return posterior_mean,posterior_variance,posterior_log_variance_cliped
    
    # compute x_0 from (x_t and predict noise): the reverse of 'q_sample'
    # DNN 输出 noise, noise的系数？？？？？？
    def predict_start_from_noise(self,x_t,t,noise):
        return self._extract(self.sqrt_recip_alphas_cumprod,t,x_t.shape)*x_t - self._extract(self.sqrt_recipml_alphas_cumprod,t,x_t.shape)*noise

    # compute predicted mean and variance of p(x_{t-1}|x_t)
    def p_mean_variance(self,model,x_t,t,clip_denoised=True):
        # predict noise using model
        pred_noise = model(x_t,t)
        # get the predicted x_0
        x_recon = self.predict_start_from_noise(x_t,t,pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon,min=-1.,max=1.)
        model_mean,posterior_variance,posterior_log_variance = self.q_posterior_mean_variance(x_recon,x_t,t)
        return model_mean,posterior_variance,posterior_log_variance

    # denoise_step:sample x_{t-1} from x_t and pred_noise
    @torch.no_grad()
    def p_sample(self,model,x_t,t,clip_denoised=True):
        #predict mean and variance
        model_mean,_,model_log_variance = self.p_mean_variance(model,x_t,t,clip_denoised=True)
        noise = torch.randn_like(x_t)
        # no noise when t==0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))####.......................
        #compute x_{t-1}
        pred_img = model_mean+nonzero_mask*(0.5*model_log_variance).exp()*noise
        return pred_img
    
    # denoise:reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self,model,shape):
        batch_size = shape[0]
        device = next(model.parameters()).device
        #start from pure noise(for each example in the batch)
        img = torch.randn(shape,device=device)
        imgs = []
        for i in tqdm(reversed(range(0,self.timesteps)),desc='sampling loop time step',total=self.timesteps):
            img = self.p_sample(model,img,torch.full((batch_size,),i,device=device,dtype=torch.long))
            imgs.append(img.cpu().numpy())
        return imgs
    
    # sampe new images
    @torch.no_grad()
    def sample(self,model,image_size,batch_size=8,channels=3):
        return self.p_sample_loop(model,shape=(batch_size,channels,image_size,image_size))
    
    # compute train loss

    def train_losses(self,model,x_start,t):
        # generate random noise
        noise = torch.randn_like(x_start)
        # get x_t,得到timestep=t时刻的噪声数据，t用来gather noise_rate和signal_rate
        x_noise = self.q_sample(x_start,t,noise=noise)  #x_start:原始图像 t:在timestep=t处，也就是加t次高斯噪声noise时候进行采样，利用到了加噪过程中的“一个重要性质”
        predicted_noise = model(x_noise,t) #预测的是x_{t-1}->x_t的噪声还是最初的噪声？？？？？预测的是‘一个重要性质’中的噪声
        loss = F.mse_loss(noise,predicted_noise)
        return loss

