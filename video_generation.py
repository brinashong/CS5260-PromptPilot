import os
import sys
import math
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'SEINE'))

import utils
from diffusion import create_diffusion

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import torchvision

from einops import rearrange
from models import get_models
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models.clip import TextEmbedder
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from torchvision import transforms
from SEINE import video_transforms
# from dataset import video_transforms
from utils import mask_generation_before
from natsort import natsorted
from diffusers.utils.import_utils import is_xformers_available
import pdb
import datetime
import gc

class Seine:
    def __init__(self, args):
        print('Initializing SEINE model...')

        if args.seed:
            torch.manual_seed(args.seed)
        torch.set_grad_enabled(False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if args.ckpt is None:
            raise ValueError("Please specify a checkpoint path using --ckpt <path>")

        # load model
        self.latent_h = args.image_size[0] // 8
        self.latent_w = args.image_size[1] // 8
        self.image_h = args.image_size[0]
        self.image_w = args.image_size[1]
        self.model = get_models(args).to(self.device)

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.model.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        ckpt_path = args.ckpt 
        state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)['ema']
        self.model.load_state_dict(state_dict)

        self.model.eval()
        pretrained_model_path = args.pretrained_model_path
        self.diffusion = create_diffusion(str(args.num_sampling_steps))
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(self.device)
        self.text_encoder = TextEmbedder(pretrained_model_path).to(self.device)
        if args.use_fp16:
            # print('Warning: using half percision for inferencing!')
            self.vae.to(dtype=torch.float16)
            self.model.to(dtype=torch.float16)
            self.text_encoder.to(dtype=torch.float16)

        self.mask_type = args.mask_type
        self.num_frames = args.num_frames
        self.use_fp16 = args.use_fp16
        self.do_classifier_free_guidance = args.do_classifier_free_guidance
        self.sample_method = args.sample_method
        self.cfg_scale = args.cfg_scale
        self.use_mask = args.use_mask

        print('Initialization complete!')

    def get_input(self, image): #, input_path):
        transform_video = transforms.Compose([
                            video_transforms.ToTensorVideo(), # TCHW
                            video_transforms.ResizeVideo((self.image_h, self.image_w)),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                        ])
        
        print("Loading the input image...")
        video_frames = []
        num = int(self.mask_type.split('first')[-1])
        first_frame = torch.as_tensor(np.array(image, dtype=np.uint8, copy=True)).unsqueeze(0)
        for i in range(num):
            video_frames.append(first_frame)
        num_zeros = self.num_frames-num
        for i in range(num_zeros):
            zeros = torch.zeros_like(first_frame)
            video_frames.append(zeros)
        n = 0
        video_frames = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2) # f,c,h,w
        video_frames = transform_video(video_frames)
        return video_frames, n

    def auto_inpainting(self, video_input, masked_video, mask, prompt, negative_prompt):
        b,f,c,h,w = video_input.shape

        del video_input
        gc.collect()
        torch.cuda.empty_cache()

        # prepare inputs
        if self.use_fp16:
            z = torch.randn(1, 4, self.num_frames, self.latent_h, self.latent_w, dtype=torch.float16, device=self.device) # b,c,f,h,w
            masked_video = masked_video.to(dtype=torch.float16)
            mask = mask.to(dtype=torch.float16)
        else:
            z = torch.randn(1, 4, self.num_frames, self.latent_h, self.latent_w, device=self.device) # b,c,f,h,w

        masked_video = rearrange(masked_video, 'b f c h w -> (b f) c h w').contiguous()
        with torch.no_grad():
            masked_video = self.vae.encode(masked_video).latent_dist.sample().mul_(0.18215)
        masked_video = rearrange(masked_video, '(b f) c h w -> b c f h w', b=b).contiguous()
        mask = torch.nn.functional.interpolate(mask[:,:,0,:], size=(self.latent_h, self.latent_w)).unsqueeze(1)
    
        # classifier_free_guidance
        if self.do_classifier_free_guidance:
            masked_video = torch.cat([masked_video] * 2)
            mask = torch.cat([mask] * 2)
            z = torch.cat([z] * 2)
            prompt_all = [prompt] + [negative_prompt]

        else:
            masked_video = masked_video
            mask = mask
            z = z
            prompt_all = [prompt]

        print(f"video prompt: {prompt_all}")

        text_prompt = self.text_encoder(text_prompts=prompt_all, train=False)
        model_kwargs = dict(encoder_hidden_states=text_prompt, 
                                class_labels=None, 
                                cfg_scale=self.cfg_scale,
                                use_fp16=self.use_fp16,) # tav unet

        # sample video
        if self.sample_method == 'ddim':
            samples = self.diffusion.ddim_sample_loop(
                self.model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=self.device, \
                mask=mask, x_start=masked_video, use_concat=self.use_mask
            )
        elif self.sample_method == 'ddpm':
            samples = self.diffusion.p_sample_loop(
                self.model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=self.device, \
                mask=mask, x_start=masked_video, use_concat=self.use_mask
            )

        del masked_video, z
        torch.cuda.empty_cache()

        samples, _ = samples.chunk(2, dim=0) # [1, 4, 16, 32, 32]
        if self.use_fp16:
            samples = samples.to(dtype=torch.float16)

        video_clip = samples[0].permute(1, 0, 2, 3).contiguous() # [16, 4, 32, 32]
        with torch.no_grad():
            video_clip = self.vae.decode(video_clip / 0.18215).sample # [16, 3, 256, 256]

        return video_clip

    def generate_video(self, save_path, image, user_prompt):
        prompt = user_prompt
        
        if not os.path.exists(os.path.join(save_path)):
            os.makedirs(os.path.join(save_path))
        video_input, reserve_frames = self.get_input(image) # f,c,h,w
        video_input = video_input.to(self.device).unsqueeze(0)  # b,f,c,h,w
        mask = mask_generation_before(self.mask_type, video_input.shape, video_input.dtype, self.device) # b,f,c,h,w
        masked_video = video_input * (mask == 0)

        video_clip = self.auto_inpainting(video_input, masked_video, mask, prompt, "") #, args.negative_prompt)
        video_ = ((video_clip * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.mp4"
        save_video_path = os.path.join(save_path, filename)
        torchvision.io.write_video(save_video_path, video_, fps=8)
        print(f'Video saved in {save_video_path}')

        return save_video_path