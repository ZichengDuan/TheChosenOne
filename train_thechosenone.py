import argparse
import itertools
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict
from torch.utils.data import Dataset
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel, CLIPTokenizer
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.lora import LoRALinearLayer, text_encoder_lora_state_dict
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
import PIL
import safetensors
import yaml
import numpy as np
from diffusers import DiffusionPipeline
from sdxl_text_inv_lora import train as train_pipeline
import shutil
from pathlib import Path
import torchvision.transforms as T
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def train_loop(args, loop_num: int, vis=True):
    """
    train and load the trained diffusion model, save the images and model file.
    """
    output_dir_base = args.output_dir
    train_data_dir_base = args.train_data_dir
    
    args.kmeans_center = int(args.num_of_generated_img / args.dsize_c)
    
    # copy dataset images from designedated folder as the init data batch
    # prepare_init_images(args.dataset_root, args.train_data_dir)
    
    # initial pair wise distance
    init_dist = 0
    
    # start looping
    for loop in range(0, loop_num):
        dinov2 = load_dinov2()
        
        # re-init the pipeline
        if loop == 0:
            pipe = load_trained_pipeline()
        else:
            # args.output_dir_per_loop = os.path.join(output_dir_base, args.character_name, str(loop - 1))
            # load from the output dir in previous loop
            pipe = load_trained_pipeline(model_path=args.output_dir_per_loop, 
                                          load_lora=True, 
                                          lora_path=os.path.join(args.output_dir_per_loop, f"checkpoint-{args.checkpointing_steps * args.num_train_epochs}"))
        
        # update model output dir for each loop
        args.output_dir_per_loop = os.path.join(output_dir_base, args.character_name, str(loop))
        
        # set up the training data folder used in training
        args.train_data_dir_per_loop = os.path.join(train_data_dir_base, args.character_name, str(loop))
        os.makedirs(args.train_data_dir_per_loop, exist_ok=True)
        
        # generate new images
        image_embs = []
        images = []
        for n_img in range(args.num_of_generated_img):
            torch.manual_seed(n_img)
            image = generate_images(pipe, prompt=args.inference_prompt, infer_steps=args.infer_steps)
            images.append(image)
            image_embs.append(infer_model(dinov2, image).detach().cpu().numpy())
        
        del pipe
        del dinov2
        torch.cuda.empty_cache()
        
        # reshaping
        embeddings = np.array(image_embs)
        embeddings = embeddings.reshape(len(image_embs), -1)
        
        # evaluate convergence
        if loop == 0:
            pairwise_distances = cdist(embeddings, embeddings, 'euclidean')
            init_dist = np.mean(pairwise_distances)
        else:
            pairwise_distances = cdist(embeddings, embeddings, 'euclidean')
            if pairwise_distances < init_dist * 0.8:
                return os.path.join(output_dir_base, args.character_name, str(loop - 1)), 
        
        # clustering
        centers, labels, elements, images = kmeans_clustering(args, embeddings, images = images)
        
        # visualize
        if vis:
            kmeans_2D_visualize(args, centers, elements, labels, loop)
        
        # evaluate
        center_norms = np.linalg.norm(centers[labels] - elements, axis=-1, keepdims=True) # each data point subtract its coresponding center
        cohesions = np.zeros(len(np.unique(labels)))
        for label_id in range(len(np.unique(labels))):
            cohesions[label_id] = sum(center_norms[labels == label_id]) / sum(labels == label_id)
            
        min_cohesion_label = np.argmin(cohesions)
        idx = np.where(labels == min_cohesion_label)[0]
        # train_samples = images[labels == min_cohesion_label]
        for sample_id, sample in enumerate(images):
            if sample_id in idx:
                sample.save(os.path.join(args.train_data_dir_per_loop, f"{sample_id}.png"))
        
        # train and save the models according to each loop's folder, and end the loop
        
        train_pipeline(args)
        
        # load the model and generate N new images
        # pipe = load_trained_pipeline(model_path=args.output_dir_per_loop, load_lora=True, lora_path=os.path.join(args.output_dir_per_loop, f"checkpoint-{args.checkpointing_steps * args.num_train_epochs}"))
        
        # # generate new images
        # for n_img in range(args.num_of_generated_img):
        #     torch.manual_seed(n_img) # set up seed
        #     image = generate_images(pipe, prompt=args.inference_prompt, infer_steps=50)
        #     image.save(os.path.join(args.train_data_dir_per_loop, args.character_name, loop, f"{n_img}.png"))



def kmeans_clustering(args, data_points, images = None):
    kmeans = KMeans(n_clusters=args.kmeans_center, init='k-means++', random_state=42)
    kmeans.fit(data_points)
    labels = kmeans.labels_

    # 统计每个聚类的数量
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))

    # 筛选数量大于10的聚类
    selected_clusters = [cluster for cluster, count in cluster_counts.items() if count > args.dmin_c]

    # 获取相应的聚类中心
    selected_centers = kmeans.cluster_centers_[selected_clusters]

    # 获取对应的标签和元素
    selected_labels = np.array([label for label in labels if label in selected_clusters])
    selected_elements = np.array([data_points[i] for i, label in enumerate(labels) if label in selected_clusters])
    if images:
        selected_images = [images[i] for i, label in enumerate(labels) if label in selected_clusters]
    else:
        selected_images = None

    return selected_centers, selected_labels, selected_elements, selected_images


def kmeans_2D_visualize(args, centers, data, labels, loop_num):
    # visualize 2D t-SNE results
    plt.figure(figsize=(20, 16))
    tsne = TSNE(n_components=2, random_state=42, perplexity=len(data) - 1)
    # centers_2d = tsne.fit_transform(centers)
    embeddings_2d = tsne.fit_transform(data)
    
    for i in range(args.kmeans_center):
        cluster_points = np.array(embeddings_2d[labels==i])
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i + 1}", s=100)
    # plt.scatter(centers[:, 0], centers_2d[:, 1], c='black', marker='x', s=200, label="Centers")
    plt.savefig(f"KMeans_res_Loop_{loop_num}.png")
    
        
def compare_features(image_features, cluster_centroid):
    # Calculate the Euclidean distance between the two feature vectors
    distance = np.linalg.norm(image_features - cluster_centroid)
    return distance


def prepare_init_images(source_path, target_root_path):
    img_out_base = target_root_path
    init_loop_img_fdr = os.path.join(img_out_base, "0")
    os.makedirs(init_loop_img_fdr, exist_ok=True)
    
    for item in os.listdir(source_path):
        src_path = os.path.join(source_path, item)
        dest_path = os.path.join(init_loop_img_fdr, item)
        
        shutil.copy2(src_path, dest_path)
        print(f"Copied {src_path} to {dest_path}")


def load_trained_pipeline(model_path = None, load_lora=True, lora_path=None):
    """
    load the diffusion pipeline according to the trained model
    """
    if model_path is not None:
        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        if load_lora:
            pipe.load_lora_weights(lora_path)
    else:
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
    pipe.to("cuda")
    return pipe


def config_2_args(path):
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    parser = argparse.ArgumentParser(description="从 YAML 配置生成 argparse 参数")
    for key, value in yaml_data.items():
        # if isinstance(value, bool):
        #     parser.add_argument(f'--{key}', action='store_true')
        # else:
            parser.add_argument(f'--{key}', type=type(value), default=value)
    
    args = parser.parse_args([])
        
    return args

def infer_model(model, image):
    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(image).unsqueeze(0).cuda()
    cls_token = model(image, is_training=False)
    return cls_token

def generate_images(pipe: StableDiffusionXLPipeline, prompt: str, infer_steps, guidance_scale=7.5):
    """
    use the given DiffusionPipeline, generate N images for the same character
    return: image, in PIL
    """
    image = pipe(prompt, num_inference_steps=infer_steps, guidance_scale=guidance_scale).images[0]
    return image


def load_dinov2():
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda()
    dinov2_vitl14.eval()
    return dinov2_vitl14


if __name__ == "__main__":
    args = config_2_args("/home/zicheng/Projects/storyTeller/config/theChosenOne.yaml")
    # load_trained_pipeline(args)
    # prepare_init_images("/home/zicheng/Data/theChosenOne/luna", "/home/zicheng/Projects/storyTeller/out_images")
    
    # model = load_dinov2()
    # infer_model(model, Image.open("/home/zicheng/Projects/storyTeller/pokemon.png"))
    # args.kmeans_center = int(args.num_of_generated_img / args.dsize_c)
    # data = torch.rand((100, 3)) * 1000
    # selected_centers, selected_labels, selected_elements = kmeans_clustering(args, data)
    # kmeans_2D_visualize(args, selected_centers, selected_elements, selected_labels, 0)
    _ = train_loop(args, 3)
    
    print(args)