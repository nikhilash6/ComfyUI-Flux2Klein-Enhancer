import math
import torch
import comfy.model_management
import comfy.utils
import latent_preview


def _time_shift(mu, sigma, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def _get_lin_function(x1=256, y1=0.5, x2=4096, y2=1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def _get_schedule(num_steps, image_seq_len, base_shift=0.5, max_shift=1.15):
    timesteps = torch.linspace(1, 0, num_steps + 1)
    mu = _get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
    result = []
    for t in timesteps:
        tv = t.item()
        if 0 < tv < 1:
            result.append(_time_shift(mu, 1.0, tv))
        else:
            result.append(tv)
    return result


class Flux2KleinKSampler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "steps": ("INT", {
                    "default": 25, "min": 1, "max": 100,
                    "tooltip": "Denoising steps. Base uses 25. Distilled uses 4-8.",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Guidance embedding value. Base uses 4.0. Distilled uses 1.0.",
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xffffffffffffffff,
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "1.0 = full denoise. Lower = partial denoise.",
                }),
                "base_shift": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Schedule base shift. Default 0.5.",
                }),
                "max_shift": ("FLOAT", {
                    "default": 1.15, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Schedule max shift. Default 1.15.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "conditioning/flux2klein"

    def sample(self, model, positive, negative, latent_image, steps=25,
               guidance_scale=4.0, seed=42, denoise=1.0,
               base_shift=0.5, max_shift=1.15):

        device = comfy.model_management.get_torch_device()
        latent = latent_image["samples"]
        B, C, H, W = latent.shape
        image_seq_len = H * W

        schedule = _get_schedule(steps, image_seq_len,
                                base_shift=base_shift, max_shift=max_shift)

        if denoise < 1.0:
            start_idx = 0
            for i, t in enumerate(schedule):
                if t <= denoise:
                    start_idx = max(0, i)
                    break
            schedule = schedule[start_idx:]

        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(latent.shape, generator=generator,
                            dtype=torch.float32, device="cpu")

        if denoise < 1.0:
            t_start = schedule[0]
            x = ((1.0 - t_start) * latent + t_start * noise).to(device)
        else:
            x = noise.to(device)

        comfy.model_management.load_model_gpu(model)
        diffusion_model = model.model.diffusion_model
        model_dtype = next(diffusion_model.parameters()).dtype
        x = x.to(device, dtype=model_dtype)

        mu = _get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        patch_count = len(model.patches) if hasattr(model, 'patches') else 0
        print(f"[Flux2Klein KSampler] patches: {patch_count} | "
              f"mu: {mu:.4f} | shift: {math.exp(mu):.2f} | "
              f"seq_len: {image_seq_len} | steps: {len(schedule)-1}")

        cond = positive[0][0].to(device, dtype=model_dtype)
        if cond.shape[0] != B:
            cond = cond[:1].expand(B, -1, -1)

        cond_meta = positive[0][1] if len(positive[0]) > 1 else {}
        ref_latents = None
        for key in ["ref_latents", "reference_latents", "concat_latent_image"]:
            if key in cond_meta:
                ref_val = cond_meta[key]
                if isinstance(ref_val, torch.Tensor):
                    ref_latents = [ref_val.to(device, dtype=model_dtype)]
                elif isinstance(ref_val, list):
                    ref_latents = [r.to(device, dtype=model_dtype) for r in ref_val]
                break

        guidance_vec = torch.full(
            (B,), guidance_scale, device=device, dtype=model_dtype
        )

        transformer_options = model.model_options.get(
            "transformer_options", {}
        ).copy()

        total_steps = len(schedule) - 1
        pbar = comfy.utils.ProgressBar(total_steps)
        preview_callback = latent_preview.prepare_callback(model, total_steps)

        with torch.no_grad():
            for i in range(total_steps):
                comfy.model_management.throw_exception_if_processing_interrupted()

                t_curr = schedule[i]
                t_prev = schedule[i + 1]

                t_vec = torch.full(
                    (B,), t_curr, device=device, dtype=model_dtype
                )
                transformer_options["sigmas"] = t_vec

                pred = diffusion_model.forward(
                    x, t_vec, cond,
                    y=None,
                    guidance=guidance_vec,
                    ref_latents=ref_latents,
                    control=None,
                    transformer_options=transformer_options,
                )

                if preview_callback is not None:
                    x0_est = (x - t_curr * pred) if t_curr > 1e-6 else x
                    preview_callback(i, x0_est.cpu().float(),
                                     x.cpu().float(), total_steps)

                x = x + (t_prev - t_curr) * pred
                pbar.update(1)

        return ({"samples": x.cpu().float()},)


NODE_CLASS_MAPPINGS = {
    "Flux2KleinKSampler": Flux2KleinKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinKSampler": "FLUX.2 Klein KSampler",
}
