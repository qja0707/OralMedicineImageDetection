import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


DEFAULT_PROMPTS = [
    "a close-up product photo of a white round pill",
    "a close-up product photo of a pink oval pill",
]


def generate_sdxl_lora_samples(
    base_model,
    lora_dir,
    output_dir,
    prompts,
    negative_prompt=None,
    num_images_per_prompt=1,
    num_inference_steps=20,
    guidance_scale=7.5,
    lora_scale=1.0,
    height=512,
    width=512,
    seed=42,
    cpu_offload=True,
):
    lora_dir = Path(lora_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (lora_dir / "pytorch_lora_weights.safetensors").exists():
        raise FileNotFoundError(f"LoRA weights not found under: {lora_dir}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.load_lora_weights(str(lora_dir))
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    saved_paths = []

    for prompt_index, prompt in enumerate(prompts):
        for image_index in range(num_images_per_prompt):
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                cross_attention_kwargs={"scale": lora_scale},
                height=height,
                width=width,
                generator=generator,
            ).images[0]

            output_path = output_dir / f"sample_{prompt_index:02d}_{image_index:02d}.png"
            image.save(output_path)
            saved_paths.append(output_path)
            print(f"saved: {output_path}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Generate sample images with a trained SDXL LoRA checkpoint.")
    parser.add_argument(
        "--base-model",
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument(
        "--lora-dir",
        default="ml/outputs/checkpoints/sdxl_lora",
    )
    parser.add_argument(
        "--output-dir",
        default="ml/outputs/samples/sdxl_lora",
    )
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--num-images-per-prompt", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cpu-offload", action="store_true")
    args = parser.parse_args()

    generate_sdxl_lora_samples(
        base_model=args.base_model,
        lora_dir=args.lora_dir,
        output_dir=args.output_dir,
        prompts=args.prompts or DEFAULT_PROMPTS,
        negative_prompt=args.negative_prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        lora_scale=args.lora_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
        cpu_offload=not args.no_cpu_offload,
    )


if __name__ == "__main__":
    main()
