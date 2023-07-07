
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline as OneStepSDPipeline
from dift.dift_ours import SDFeaturizer

import copy
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as FF

from PIL import Image
from utils import save_image_with_points, motion_supervison


sd_id = 'runwayml/stable-diffusion-v1-5'

unet = UNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")

gc.collect()
onestep_pipe = onestep_pipe.to("cuda")
onestep_pipe.enable_attention_slicing()
onestep_pipe.enable_xformers_memory_efficient_attention()
pipe = onestep_pipe
dift = SDFeaturizer()
pipe.load_lora_weights('./lora-200')
dift.pipe.load_lora_weights('./lora-200')

bs = 1

# Trials required. 
# Also differs from the paper.
prompt = "xxy5syt00"
ddim_step = 50
guidance_scale = 0
inter_id = 181
max_iters = 9
r1 = 7
r2 = 15
lam = 0.1
reg = 0
lr = 4e-2
up_ft_index = 2

# control points & mask
handle_points = [[275, 283]]
target_points = [[265, 268]]
handle_points0 = copy.deepcopy(handle_points)
n = len(handle_points)
mask = torch.zeros((1, 1, 512, 512)).to("cuda")
mask[..., 265-180:275+200, 268-240:283+60] = 1

with torch.no_grad():
    prompt_embeds = pipe._encode_prompt(
        prompt=prompt,
        device='cuda',
        num_images_per_prompt=1,
        do_classifier_free_guidance=True)

image0 = Image.open("finetune_data/bear.jpg")
image0 = np.array(image0).astype(np.float32) / 255 * 2 - 1
image0 = torch.from_numpy(image0).permute(2, 0, 1).unsqueeze(0)

pipe.scheduler.set_timesteps(ddim_step)
resume_times = pipe.scheduler.timesteps[list(pipe.scheduler.timesteps).index(inter_id)+1:]

with torch.no_grad():
    intermediate = pipe.vae.encode(image0.cuda()).latent_dist.sample() * pipe.vae.config.scaling_factor
    noise = torch.randn_like(intermediate).cuda()
    intermediate =  pipe.scheduler.add_noise(intermediate, noise, torch.tensor(inter_id, dtype=torch.long).cuda())

    latent_model_input = torch.cat([intermediate] * 2)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, inter_id)

    # predict the noise residual
    noise_pred = pipe.unet(
        latent_model_input,
        inter_id,
        encoder_hidden_states=prompt_embeds
    ).sample

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents_0_t_minus_1 = pipe.scheduler.step(noise_pred, inter_id, intermediate).prev_sample
    latents_0_t_minus_1 = nn.Upsample(size=(image0.shape[2], image0.shape[3]), mode='bilinear')(latents_0_t_minus_1)

with torch.no_grad():

    F00 = dift.latent_feature(intermediate.float(),
                    prompt=prompt,
                    t=inter_id,
                    up_ft_index=up_ft_index,
                    ensemble_size=1)

    F0 = dift.forward(image0,
                      prompt=prompt,
                      t=261,
                      up_ft_index=up_ft_index,
                      ensemble_size=1)
    
    F00 = nn.Upsample(size=(image0.shape[2], image0.shape[3]), mode='bilinear')(F00)
    F0 = nn.Upsample(size=(image0.shape[2], image0.shape[3]), mode='bilinear')(F0)


learnable_param = intermediate.clone().requires_grad_(True)
optimizer = torch.optim.Adam([learnable_param], lr=lr)
for iter in range(max_iters):
    optimizer.zero_grad()

    latents = learnable_param

    # motion supervision
    F2 = dift.latent_feature(latents.float(),
                      prompt=prompt,
                      t=inter_id,
                      up_ft_index=up_ft_index,
                      ensemble_size=1)
    F2 = nn.Upsample(size=(image0.shape[2], image0.shape[3]), mode='bilinear')(F2)

    loss = motion_supervison(handle_points, target_points, F2, r1)
    if mask is not None:
        latent_model_input = latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, inter_id)

        # predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input,
            inter_id,
            encoder_hidden_states=prompt_embeds.chunk(2)[0]
        ).sample

        latents_t_minus_1 = pipe.scheduler.step(noise_pred, inter_id, latents).prev_sample
        latents_t_minus_1 = nn.Upsample(size=(image0.shape[2], image0.shape[3]), mode='bilinear')(latents_t_minus_1)

        mask_loss = ((latents_t_minus_1 - latents_0_t_minus_1) * (1-mask)).abs().mean()
        loss += mask_loss * lam

    loss += FF.l1_loss(latents, intermediate) * reg

    loss.backward()
    optimizer.step()

    del F2
    gc.collect()
    torch.cuda.empty_cache()

    latents = learnable_param.clone().detach()
    with torch.no_grad():
        for t in resume_times:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # point tracking
    with torch.no_grad():
        latents = 1 / pipe.vae.config.scaling_factor * latents
        image = pipe.vae.decode(latents).sample
    
        F2 = dift.forward(image.float(),
                        prompt=prompt,
                        t=261,
                        up_ft_index=up_ft_index,
                        ensemble_size=1)
        
        F2 = nn.Upsample(size=(image.shape[2], image.shape[3]), mode='bilinear')(F2)

        for i in range(n):
            # pi = handle_points0[i]
            pi = handle_points[i]
            pi = torch.tensor(pi)

            up = int(max(pi[0] - r2, 0))
            down = int(min(pi[0] + r2 + 1, 512))
            left = int(max(pi[1] - r2, 0))
            right = int(min(pi[1] + r2 + 1, 512))
            feat_patch = F2[:,:,up:down,left:right]

            # L2 = torch.linalg.norm(feat_patch - F0[:,:,int(pi[0]),int(pi[1])].reshape(1,-1,1,1), dim=1)
            L2 = torch.linalg.norm(feat_patch - F0[:,:,handle_points0[i][0],handle_points0[i][1]].reshape(1,-1,1,1), dim=1)

            _, idx = torch.min(L2.view(1,-1), -1)
            width = right - left
            point = [idx.item() // width + up, idx.item() % width + left]
            handle_points[i] = point

        save_image_with_points(image, point[0], point[1], "bear_iter_{}".format(iter))

    del F2
    gc.collect()
    torch.cuda.empty_cache()

    print("iter: {}, loss: {}, handle points: {}, target points: {}".format(iter, 
                loss.item(), handle_points, target_points))


latents = learnable_param.detach()
with torch.no_grad():
    for t in pipe.progress_bar(resume_times):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

with torch.no_grad():
    latents = 1 / pipe.vae.config.scaling_factor * latents
    image = pipe.vae.decode(latents).sample

image = (image / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()
image_save = Image.fromarray((image[0] * 255).astype(np.uint8))
image_save.save("images/bear_output.png")
