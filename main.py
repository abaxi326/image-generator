import base64
import importlib
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "stabilityai/stable-diffusion-xl-base-1.0")

# Only the pipelines we explicitly support — avoids importing broken optional deps
PIPELINE_MAP = {
    "StableDiffusionPipeline":      "diffusers.StableDiffusionPipeline",
    "StableDiffusionXLPipeline":    "diffusers.StableDiffusionXLPipeline",
    "FluxPipeline":                 "diffusers.FluxPipeline",
    "StableDiffusion3Pipeline":     "diffusers.StableDiffusion3Pipeline",
    "PixArtAlphaPipeline":          "diffusers.PixArtAlphaPipeline",
    "PixArtSigmaPipeline":          "diffusers.PixArtSigmaPipeline",
    "KandinskyV22Pipeline":         "diffusers.KandinskyV22Pipeline",
    "WuerstchenDecoderPipeline":    "diffusers.WuerstchenDecoderPipeline",
}

pipeline = None
model_info: dict = {}


def _load_pipeline_class(model_path: str):
    """Read model_index.json and return the matching pipeline class."""
    config_file = Path(model_path) / "model_index.json"
    if not config_file.exists():
        raise RuntimeError(f"model_index.json not found in {model_path}")

    with open(config_file) as f:
        config = json.load(f)

    class_name = config.get("_class_name", "")
    if class_name not in PIPELINE_MAP:
        raise RuntimeError(
            f"Pipeline '{class_name}' is not in the supported list: {list(PIPELINE_MAP)}"
        )

    module_name, attr = PIPELINE_MAP[class_name].rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr), class_name


def _resolve_dtype(class_name: str) -> torch.dtype:
    return torch.bfloat16 if "Flux" in class_name else torch.float16


def _supports_negative_prompt(class_name: str) -> bool:
    return "Flux" not in class_name


def _generator_device(class_name: str) -> str:
    return "cpu" if "Flux" in class_name else "cuda"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, model_info
    PipelineClass, class_name = _load_pipeline_class(MODEL_PATH)
    dtype = _resolve_dtype(class_name)
    logger.info(f"Loading '{MODEL_PATH}' as {class_name} with dtype={dtype} ...")
    t0 = time.time()

    pipeline = PipelineClass.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
    )
    pipeline.enable_model_cpu_offload()

    model_info = {
        "model_path": MODEL_PATH,
        "pipeline_class": class_name,
        "supports_negative_prompt": _supports_negative_prompt(class_name),
        "dtype": str(dtype).replace("torch.", ""),
    }
    logger.info(f"Loaded {class_name} in {time.time() - t0:.1f}s")
    yield
    del pipeline
    torch.cuda.empty_cache()


app = FastAPI(title="Image Generator", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    num_inference_steps: int = Field(default=30, ge=1, le=150)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=20.0)
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    image_base64: str
    seed: int
    elapsed_seconds: float


@app.get("/info")
async def info():
    return model_info


@app.get("/health")
async def health():
    return {"status": "ready" if pipeline is not None else "loading", "cuda": torch.cuda.is_available()}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    seed = req.seed if req.seed is not None else torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator(device=_generator_device(model_info["pipeline_class"])).manual_seed(seed)

    logger.info(f"Generating: '{req.prompt[:80]}' seed={seed}")
    t0 = time.time()

    kwargs = dict(
        prompt=req.prompt,
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        generator=generator,
    )
    if model_info.get("supports_negative_prompt") and req.negative_prompt:
        kwargs["negative_prompt"] = req.negative_prompt

    result = pipeline(**kwargs)
    elapsed = round(time.time() - t0, 2)
    logger.info(f"Generated in {elapsed}s")

    buf = io.BytesIO()
    result.images[0].save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return GenerateResponse(image_base64=b64, seed=seed, elapsed_seconds=elapsed)


static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
