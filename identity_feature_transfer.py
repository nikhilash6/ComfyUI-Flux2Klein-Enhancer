"""Identity Feature Transfer — Attention Output Steering."""

import torch
import torch.nn.functional as F


class IdentityFeatureTransfer:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Requires ReferenceLatent connected. The reference must be in the image stream.",
                }),
                "strength": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Per-block blend factor. Fires at every active block so the effect is cumulative. Start at 0.10 to 0.20.",
                }),
                "start_block": ("INT", {
                    "default": 0, "min": 0, "max": 23,
                    "tooltip": "First block index to apply. 0 = earliest. Index is shared across double and single blocks (resets when single blocks begin).",
                }),
                "end_block": ("INT", {
                    "default": 23, "min": 0, "max": 23,
                    "tooltip": "Last block index to apply. Covers 8 double blocks (0-7) then 24 single blocks (index resets 0-23). Higher values extend coverage into later single blocks.",
                }),
                "mode": (["cosine_pull", "topk_replace", "mean_transfer"], {
                    "default": "cosine_pull",
                    "tooltip": "cosine_pull: pulls each gen token toward its best-matching ref token. topk_replace: only affects the top K%% most similar tokens. mean_transfer: shifts overall feature distribution toward the reference.",
                }),
                "top_k_percent": ("FLOAT", {
                    "default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "topk_replace mode only. Fraction of generation tokens to affect. 0.25 = top 25%% most similar.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2klein"

    def apply(self, model, strength=0.15, start_block=0, end_block=23,
              mode="cosine_pull", top_k_percent=0.25):
        m = model.clone()

        _strength = strength
        _start = start_block
        _end = end_block
        _mode = mode
        _topk_pct = top_k_percent

        def output_patch(attn, extra_options):
            ref_tokens_list = extra_options.get("reference_image_num_tokens", [])
            if not ref_tokens_list:
                return attn

            block_idx = extra_options.get("block_index", 0)
            if block_idx < _start or block_idx > _end:
                return attn

            img_slice = extra_options.get("img_slice", None)
            if img_slice is None:
                return attn

            txt_end = img_slice[0]
            total_seq = img_slice[1]

            total_ref = sum(ref_tokens_list)
            if total_ref <= 0:
                return attn

            gen_start = txt_end
            gen_end = total_seq - total_ref
            ref_start = total_seq - total_ref
            ref_end = total_seq

            if gen_end <= gen_start or ref_end <= ref_start:
                return attn

            gen_features = attn[:, gen_start:gen_end]
            ref_features = attn[:, ref_start:ref_end]

            if _mode == "cosine_pull":
                gen_norm = F.normalize(gen_features.float(), dim=-1)
                ref_norm = F.normalize(ref_features.float(), dim=-1)

                sim = torch.bmm(gen_norm, ref_norm.transpose(1, 2))
                max_sim, max_idx = sim.max(dim=-1)

                weight = max_sim.clamp(0.0, 1.0) * _strength
                weight = weight.unsqueeze(-1).to(attn.dtype)

                max_idx_expanded = max_idx.unsqueeze(-1).expand(-1, -1, ref_features.shape[-1])
                best_ref = torch.gather(ref_features, 1, max_idx_expanded)

                new_gen = gen_features + (best_ref - gen_features) * weight

                attn = attn.clone()
                attn[:, gen_start:gen_end] = new_gen

            elif _mode == "topk_replace":
                gen_norm = F.normalize(gen_features.float(), dim=-1)
                ref_norm = F.normalize(ref_features.float(), dim=-1)
                sim = torch.bmm(gen_norm, ref_norm.transpose(1, 2))
                max_sim, max_idx = sim.max(dim=-1)

                k = max(1, int(gen_features.shape[1] * _topk_pct))
                topk_vals, topk_indices = max_sim.topk(k, dim=-1)

                max_idx_expanded = max_idx.unsqueeze(-1).expand(-1, -1, ref_features.shape[-1])
                best_ref = torch.gather(ref_features, 1, max_idx_expanded)

                attn = attn.clone()
                for b in range(attn.shape[0]):
                    for i in range(k):
                        idx = topk_indices[b, i].item()
                        sim_val = topk_vals[b, i].item()
                        if sim_val > 0:
                            w = min(sim_val * _strength, 1.0)
                            pos = gen_start + idx
                            attn[b, pos] = (1.0 - w) * attn[b, pos] + w * best_ref[b, idx]

            elif _mode == "mean_transfer":
                gen_mean = gen_features.mean(dim=1, keepdim=True)
                ref_mean = ref_features.mean(dim=1, keepdim=True)

                delta = (ref_mean - gen_mean) * _strength
                new_gen = gen_features + delta

                attn = attn.clone()
                attn[:, gen_start:gen_end] = new_gen

            return attn

        m.set_model_attn1_output_patch(output_patch)
        return (m,)




class IdentityFeatureTransferAdvanced:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Requires ReferenceLatent connected. The reference must be in the image stream.",
                }),
                "reference_index": ("INT", {
                    "default": 0, "min": 0, "max": 15, "step": 1,
                    "tooltip": "Which reference image to draw features from when multiple are connected (0 = first).",
                }),
                "mode": (["cosine_pull", "topk_replace", "mean_transfer"], {
                    "default": "cosine_pull",
                    "tooltip": "cosine_pull: each generation token is pulled toward similar reference tokens. topk_replace: only the top K%% most similar tokens are affected. mean_transfer: shifts the overall feature distribution toward the reference.",
                }),
                "top_k_percent": ("FLOAT", {
                    "default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "topk_replace mode only. Fraction of generation tokens to affect.",
                }),
                "double_enable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply transfer on double blocks (0-7). These shape pose, color, and identity early in the network.",
                }),
                "double_strength": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Per-block blend factor for double blocks. Cumulative across blocks. Raise for stronger identity guidance, especially when the reference contains multiple subjects.",
                }),
                "double_start": ("INT", {
                    "default": 0, "min": 0, "max": 7,
                    "tooltip": "First double block to apply on (0-7).",
                }),
                "double_end": ("INT", {
                    "default": 7, "min": 0, "max": 7,
                    "tooltip": "Last double block to apply on (0-7).",
                }),
                "single_enable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply transfer on single blocks (0-23). These refine style and texture later in the network.",
                }),
                "single_strength": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Per-block blend factor for single blocks. Cumulative across blocks.",
                }),
                "single_start": ("INT", {
                    "default": 0, "min": 0, "max": 23,
                    "tooltip": "First single block to apply on (0-23).",
                }),
                "single_end": ("INT", {
                    "default": 23, "min": 0, "max": 23,
                    "tooltip": "Last single block to apply on (0-23).",
                }),
                "block_schedule": (["flat", "ramp_down", "ramp_up", "peak_mid"], {
                    "default": "flat",
                    "tooltip": "Strength curve across the active block range. flat = constant. ramp_down = stronger on early blocks. ramp_up = stronger on later blocks. peak_mid = strongest in the middle.",
                }),
                "sim_floor": ("FLOAT", {
                    "default": 0.20, "min": 0.0, "max": 0.95, "step": 0.01,
                    "tooltip": "Cosine similarity threshold gating which reference-to-generation matches contribute. Low (~0.05) = wide pull, tight identity lock, suited to subtle edits like outfit swaps. High = sparse pull, more freedom for broader edits.",
                }),
                "mask_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Used only when subject_mask is connected. Reference tokens whose pooled mask value falls below this are excluded from the pull. 0.5 keeps boundary tokens; raise toward 1.0 to shrink the effective mask inward.",
                }),
            },
            "optional": {
                "subject_mask": ("MASK", {
                    "tooltip": "Optional subject mask for the reference image. When connected, the cosine pull samples only from masked-in reference tokens, leaving everything else out of the transfer. The conditioning latent is not modified, so the model still sees the full reference. Mask aspect must match the encoded reference aspect.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2klein"

    def apply(self, model, reference_index=0, mode="cosine_pull", top_k_percent=0.25,
              double_enable=True, double_strength=0.15, double_start=0, double_end=7,
              single_enable=True, single_strength=0.15, single_start=0, single_end=23,
              block_schedule="flat", sim_floor=0.20, mask_threshold=0.5,
              subject_mask=None):
        m = model.clone()

        _ref_idx = reference_index
        _mode = mode
        _topk_pct = top_k_percent
        _d_enable = double_enable
        _d_strength = double_strength
        _d_start = double_start
        _d_end = double_end
        _s_enable = single_enable
        _s_strength = single_strength
        _s_start = single_start
        _s_end = single_end
        _schedule = block_schedule
        _sim_floor = float(sim_floor)
        _mask_thresh = float(mask_threshold)

        if subject_mask is not None:
            mk = subject_mask
            if mk.dim() == 3:
                mk = mk[0]
            elif mk.dim() == 4:
                mk = mk[0, 0]
            _src_mask = mk.detach().float().cpu()
        else:
            _src_mask = None

        _idx_cache = {}

        TEMPERATURE = 0.07

        def _subject_indices(Nr, device):
            if _src_mask is None:
                return None
            if Nr in _idx_cache:
                idx = _idx_cache[Nr]
                return idx.to(device) if idx is not None else None

            mh, mw = _src_mask.shape[-2], _src_mask.shape[-1]
            target = mh / max(mw, 1)
            best = (1, Nr)
            best_err = float("inf")
            limit = int(Nr ** 0.5) + 2
            for h in range(1, limit):
                if Nr % h == 0:
                    w = Nr // h
                    for hh, ww in ((h, w), (w, h)):
                        err = abs(hh / max(ww, 1) - target)
                        if err < best_err:
                            best_err = err
                            best = (hh, ww)
            h_ref, w_ref = best

            mask_2d = _src_mask.unsqueeze(0).unsqueeze(0)
            pooled = F.adaptive_avg_pool2d(mask_2d, (h_ref, w_ref))
            flat = pooled.view(-1)
            keep = flat >= _mask_thresh
            if keep.sum().item() == 0:
                _idx_cache[Nr] = None
                return None
            idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
            _idx_cache[Nr] = idx
            return idx.to(device)

        def _schedule_multiplier(block_idx, start, end):
            if end <= start:
                return 1.0
            t = (block_idx - start) / max(end - start, 1)
            if _schedule == "flat":
                return 1.0
            elif _schedule == "ramp_down":
                return 1.0 - t
            elif _schedule == "ramp_up":
                return t
            elif _schedule == "peak_mid":
                return 1.0 - abs(2.0 * t - 1.0)
            return 1.0

        def _apply_steering(gen_features, ref_features, strength, attn, gen_start, gen_end):
            if _mode == "cosine_pull":
                gen_f = gen_features.float()
                ref_f = ref_features.float()

                full_Nr = ref_f.shape[1]
                subj_idx = _subject_indices(full_Nr, ref_f.device)
                if subj_idx is not None:
                    ref_f = ref_f.index_select(1, subj_idx)

                gen_c = gen_f - gen_f.mean(dim=1, keepdim=True)
                ref_c = ref_f - ref_f.mean(dim=1, keepdim=True)

                gen_norm = F.normalize(gen_c, dim=-1)
                ref_norm = F.normalize(ref_c, dim=-1)
                sim = torch.bmm(gen_norm, ref_norm.transpose(1, 2))

                B, Ng, Nr = sim.shape
                neg_inf = torch.finfo(sim.dtype).min

                if subj_idx is None and Ng == Nr:
                    diag = torch.arange(Ng, device=sim.device)
                    sim[:, diag, diag] = neg_inf

                sim = torch.where(sim >= _sim_floor, sim,
                                  torch.full_like(sim, neg_inf))

                attn_w = torch.softmax(sim / TEMPERATURE, dim=-1)
                attn_w = torch.nan_to_num(attn_w, nan=0.0)

                pooled_ref = torch.bmm(attn_w, ref_f)

                max_sim = sim.max(dim=-1).values
                max_sim = torch.where(torch.isinf(max_sim),
                                      torch.zeros_like(max_sim), max_sim)
                denom = max(1.0 - _sim_floor, 1e-6)
                conf = ((max_sim - _sim_floor) / denom).clamp(0.0, 1.0)
                weight = (conf * strength).unsqueeze(-1).to(attn.dtype)

                new_gen = gen_features + (pooled_ref.to(attn.dtype) - gen_features) * weight
                attn = attn.clone()
                attn[:, gen_start:gen_end] = new_gen

            elif _mode == "topk_replace":
                gen_norm = F.normalize(gen_features.float(), dim=-1)
                ref_norm = F.normalize(ref_features.float(), dim=-1)
                sim = torch.bmm(gen_norm, ref_norm.transpose(1, 2))
                max_sim, max_idx = sim.max(dim=-1)

                k = max(1, int(gen_features.shape[1] * _topk_pct))
                topk_vals, topk_indices = max_sim.topk(k, dim=-1)

                max_idx_expanded = max_idx.unsqueeze(-1).expand(-1, -1, ref_features.shape[-1])
                best_ref = torch.gather(ref_features, 1, max_idx_expanded)

                attn = attn.clone()
                for b in range(attn.shape[0]):
                    for i in range(k):
                        idx = topk_indices[b, i].item()
                        sim_val = topk_vals[b, i].item()
                        if sim_val > 0:
                            w = min(sim_val * strength, 1.0)
                            pos = gen_start + idx
                            attn[b, pos] = (1.0 - w) * attn[b, pos] + w * best_ref[b, idx]

            elif _mode == "mean_transfer":
                gen_mean = gen_features.mean(dim=1, keepdim=True)
                ref_mean = ref_features.mean(dim=1, keepdim=True)
                delta = (ref_mean - gen_mean) * strength
                new_gen = gen_features + delta
                attn = attn.clone()
                attn[:, gen_start:gen_end] = new_gen

            return attn

        def output_patch(attn, extra_options):
            ref_tokens_list = extra_options.get("reference_image_num_tokens", [])
            if not ref_tokens_list:
                return attn

            block_type = extra_options.get("block_type", "double")
            block_idx = extra_options.get("block_index", 0)

            if block_type == "double":
                if not _d_enable:
                    return attn
                if block_idx < _d_start or block_idx > _d_end:
                    return attn
                base_strength = _d_strength
                sched_mult = _schedule_multiplier(block_idx, _d_start, _d_end)
            elif block_type == "single":
                if not _s_enable:
                    return attn
                if block_idx < _s_start or block_idx > _s_end:
                    return attn
                base_strength = _s_strength
                sched_mult = _schedule_multiplier(block_idx, _s_start, _s_end)
            else:
                return attn

            strength = base_strength * sched_mult
            if strength <= 0.0:
                return attn

            img_slice = extra_options.get("img_slice", None)
            if img_slice is None:
                return attn

            txt_end = img_slice[0]
            total_seq = img_slice[1]
            total_ref = sum(ref_tokens_list)
            if total_ref <= 0:
                return attn

            gen_start = txt_end
            gen_end = total_seq - total_ref

            if gen_end <= gen_start:
                return attn

            clamped_idx = min(_ref_idx, len(ref_tokens_list) - 1)
            ref_offset = sum(ref_tokens_list[:clamped_idx])
            ref_token_count = ref_tokens_list[clamped_idx]
            ref_start = (total_seq - total_ref) + ref_offset
            ref_end = ref_start + ref_token_count

            if ref_end > total_seq or ref_token_count <= 0:
                return attn

            gen_features = attn[:, gen_start:gen_end]
            ref_features = attn[:, ref_start:ref_end]

            attn = _apply_steering(gen_features, ref_features, strength,
                                   attn, gen_start, gen_end)
            return attn

        m.set_model_attn1_output_patch(output_patch)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "IdentityFeatureTransfer": IdentityFeatureTransfer,
    "IdentityFeatureTransferAdvanced": IdentityFeatureTransferAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IdentityFeatureTransfer": "FLUX.2 Klein Identity Feature Transfer",
    "IdentityFeatureTransferAdvanced": "FLUX.2 Klein Identity Feature Transfer Advanced",
}
