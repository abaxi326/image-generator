import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel


class ImageCaptionDataset(Dataset):
    def __init__(self, data_dir: str, size: int = 1024):
        self.data_dir = Path(data_dir)
        self.size = size
        self.transform = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        exts = {".jpg", ".jpeg", ".png", ".webp"}
        self.images = sorted([p for p in self.data_dir.iterdir() if p.suffix.lower() in exts])
        if not self.images:
            raise ValueError(f"No images found in {data_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        caption_path = image_path.with_suffix(".txt")
        caption = caption_path.read_text(encoding="utf-8").strip() if caption_path.exists() else ""

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return {
            "pixel_values": image,
            "caption": caption,
            "original_size": (self.size, self.size),
            "crop_coords_top_left": (0, 0),
            "target_size": (self.size, self.size),
        }


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.lora_down = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_up = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.normal_(self.lora_down, std=1.0 / rank)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x, self.lora_down), self.lora_up)
        return base_out + lora_out * self.scale


def set_module_by_name(root: nn.Module, name: str, new_module: nn.Module):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def inject_lora_layers(unet: nn.Module, rank: int, alpha: float):
    replaced = []
    target_suffixes = ("to_q", "to_k", "to_v", "to_out.0")

    for module_name, module in list(unet.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(module_name.endswith(suffix) for suffix in target_suffixes):
            continue
        set_module_by_name(unet, module_name, LoRALinear(module, rank=rank, alpha=alpha))
        replaced.append(module_name)

    if not replaced:
        raise RuntimeError("No target linear layers found for LoRA injection.")
    return replaced


def tokenize_captions(captions, tokenizer_one, tokenizer_two, device):
    tok_one = tokenizer_one(
        captions,
        padding="max_length",
        truncation=True,
        max_length=tokenizer_one.model_max_length,
        return_tensors="pt",
    )
    tok_two = tokenizer_two(
        captions,
        padding="max_length",
        truncation=True,
        max_length=tokenizer_two.model_max_length,
        return_tensors="pt",
    )
    return tok_one.input_ids.to(device), tok_two.input_ids.to(device)


def encode_prompt(captions, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, device):
    input_ids_one, input_ids_two = tokenize_captions(captions, tokenizer_one, tokenizer_two, device)

    with torch.no_grad():
        out_one = text_encoder_one(input_ids_one, output_hidden_states=True)
        out_two = text_encoder_two(input_ids_two, output_hidden_states=True)

    prompt_embeds_one = out_one.hidden_states[-2]
    prompt_embeds_two = out_two.hidden_states[-2]
    prompt_embeds = torch.cat([prompt_embeds_one, prompt_embeds_two], dim=-1)
    pooled_prompt_embeds = out_two.text_embeds
    return prompt_embeds, pooled_prompt_embeds


def compute_time_ids(batch, device):
    time_ids = []
    for original_size, crop_xy, target_size in zip(
        batch["original_size"], batch["crop_coords_top_left"], batch["target_size"]
    ):
        values = list(original_size) + list(crop_xy) + list(target_size)
        time_ids.append(values)
    return torch.tensor(time_ids, device=device, dtype=torch.float32)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    captions = [example["caption"] for example in examples]
    original_sizes = [example["original_size"] for example in examples]
    crop_coords = [example["crop_coords_top_left"] for example in examples]
    target_sizes = [example["target_size"] for example in examples]
    return {
        "pixel_values": pixel_values,
        "caption": captions,
        "original_size": original_sizes,
        "crop_coords_top_left": crop_coords,
        "target_size": target_sizes,
    }


def save_lora_weights(unet: nn.Module, output_path: str):
    state = {}
    for name, module in unet.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_down"] = module.lora_down.detach().cpu()
            state[f"{name}.lora_up"] = module.lora_up.detach().cpu()
            state[f"{name}.alpha"] = torch.tensor(module.alpha)
    torch.save(state, output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="sdxl_lora_manual.pt")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def train_lora(args, progress_callback=None):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer_2")
    text_encoder_one = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(device)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_path, subfolder="text_encoder_2"
    ).to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device, dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet").to(
        device, dtype=weight_dtype
    )
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")

    text_encoder_one.eval()
    text_encoder_two.eval()
    vae.eval()
    unet.train()

    for module in [text_encoder_one, text_encoder_two, vae, unet]:
        for param in module.parameters():
            param.requires_grad = False

    replaced = inject_lora_layers(unet, rank=args.rank, alpha=args.alpha)
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    dataset = ImageCaptionDataset(args.train_data_dir, size=args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    total_steps = len(dataloader) * args.epochs
    intro_lines = [
        f"Injected LoRA into {len(replaced)} layers",
        f"Training on {len(dataset)} images",
        f"Using device={device.type} total_steps={total_steps}",
    ]
    for line in intro_lines:
        print(line)
        if progress_callback is not None:
            progress_callback(line)

    global_step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
            captions = batch["caption"]

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                captions, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, device
            )
            prompt_embeds = prompt_embeds.to(device=device, dtype=weight_dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=weight_dtype)
            time_ids = compute_time_ids(batch, device=device).to(dtype=weight_dtype)

            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": time_ids},
            ).sample

            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % args.log_every == 0:
                line = f"epoch={epoch + 1}/{args.epochs} step={global_step}/{total_steps} loss={loss.item():.6f}"
                print(line)
                if progress_callback is not None:
                    progress_callback(line)

    save_lora_weights(unet, args.output_path)
    line = f"Saved LoRA weights to {args.output_path}"
    print(line)
    if progress_callback is not None:
        progress_callback(line)


def build_args(**kwargs):
    defaults = {
        "pretrained_model_path": "",
        "train_data_dir": "",
        "output_path": "sdxl_lora_manual.pt",
        "resolution": 1024,
        "batch_size": 1,
        "epochs": 1,
        "lr": 1e-4,
        "rank": 4,
        "alpha": 4.0,
        "device": "cuda",
        "log_every": 10,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def main():
    args = parse_args()
    train_lora(args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
