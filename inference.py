from diffusers import DiffusionPipeline
import torch
import argparse
import yaml
import os

def config_2_args(path):
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    parser = argparse.ArgumentParser(description="Generate args from yaml")
    for key, value in yaml_data.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    
    args = parser.parse_args([])
    return args

args = config_2_args("/home/zicheng/Projects/The_Chosen_One/config/theChosenOne.yaml")

loop = 1
model_path = os.path.join(args.output_dir, args.character_name, str(loop))
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")
pipe.load_lora_weights(os.path.join(model_path, f"checkpoint-{args.checkpointing_steps * args.num_train_epochs}"))

# remember to use the place holader here
prompt = f"A photo of {args.placeholder_token} near the Statue of Liberty."
image = pipe(prompt, num_inference_steps=50, guidance_scale=15).images[0]
image.save("luna.png")