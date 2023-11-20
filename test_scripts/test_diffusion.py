from diffusers import DiffusionPipeline,DDPMScheduler
import torch
from PIL import Image,ImageEnhance
import torchvision.transforms as T
from tqdm import auto
import random

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.bfloat16,
    variant="fp16", 
    use_safetensors=False,
    add_watermarker=False,
    # use DDPM DDPMScheduler instead of default EulerDiscreteScheduler 
    scheduler = DDPMScheduler(num_train_timesteps=1000,prediction_type="epsilon",beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False)
)

base.disable_xformers_memory_efficient_attention()
torch.set_grad_enabled(True)
base=base.to("cuda")

text_encoder_1 = base.text_encoder
text_encoder_2 = base.text_encoder_2

tokenizer_1 = base.tokenizer_1
tokenizer_2 = base.tokenizer_2

vae = base.vae
unet = base.unet

print()