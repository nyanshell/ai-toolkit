
import argparse
import os
import glob
from datetime import datetime
from safetensors.torch import load_file
import torch

from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel
from toolkit.config_modules import ModelConfig, GenerateImageConfig

def apply_lora(model, lora_path, scale=1.0):
    print(f"Loading LoRA from {lora_path}")
    state_dict = load_file(lora_path)

    loaded_count = 0

    for key, weight in state_dict.items():
        if "lora_A" in key:
            # Found A, look for B
            key_B = key.replace("lora_A", "lora_B")
            if key_B not in state_dict:
                continue

            weight_B = state_dict[key_B]

            # Module name logic
            # keys are usually like transformer.double_blocks.0.img_attn.qkv.lora_A.weight
            # ZImage keys might need adjustment if they start with diffusion_model
            module_name = key.replace(".lora_A.weight", "")
            target_name = module_name

            # Remove prefix corresponding to the wrapper if present in keys
            if target_name.startswith("transformer."):
                 target_name = target_name.replace("transformer.", "")
            elif target_name.startswith("diffusion_model."):
                 target_name = target_name.replace("diffusion_model.", "")

            # Find module
            try:
                module = model
                parts = target_name.split(".")
                for part in parts:
                    module = getattr(module, part)

                if isinstance(module, torch.nn.Linear):
                    # Check shapes to be sure
                    # Linear: (out, in)
                    # A: (rank, in) B: (out, rank)
                    # B@A: (out, in)

                    if weight.shape[1] != module.weight.shape[1] or weight_B.shape[0] != module.weight.shape[0]:
                        print(f"Skipping {target_name} due to shape mismatch. Model: {module.weight.shape}, LoRA A: {weight.shape}, B: {weight_B.shape}")
                        continue

                    # Alpha
                    alpha_key = key.replace("lora_A.weight", "alpha")
                    alpha = state_dict.get(alpha_key, None)
                    rank = weight.shape[0]

                    scaling = 1.0
                    if alpha is not None:
                        scaling = alpha.item() / rank

                    final_scale = scaling * scale

                    weight = weight.to(module.weight.device, dtype=module.weight.dtype)
                    weight_B = weight_B.to(module.weight.device, dtype=module.weight.dtype)

                    with torch.no_grad():
                         module.weight += (weight_B @ weight) * final_scale

                    loaded_count += 1
            except AttributeError:
                # print(f"Module {target_name} not found")
                pass

    print(f"Applied LoRA to {loaded_count} modules")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=str, required=True, help="Path to LoRA safetensors file")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape", help="Prompt text, path to a text file, or path to a directory of text files")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--transformer_path", type=str, required=True, help="Path to ZImage model or checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--offload", action="store_true", help="Enable CPU offloading to save VRAM")
    parser.add_argument("--count", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=None, help="Seed for generation")

    args = parser.parse_args()

    prompts = []
    if os.path.isdir(args.prompt):
        print(f"Scanning directory: {args.prompt}")
        files = glob.glob(os.path.join(args.prompt, "*"))
        # filter for text files
        files = [f for f in files if f.endswith('.txt') or f.endswith('.caption')]
        files.sort()
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        prompts.append((os.path.basename(file_path), content))
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        print(f"Found {len(prompts)} prompt files")
    elif args.prompt.endswith('.txt') and os.path.exists(args.prompt):
        with open(args.prompt, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            prompts.append((os.path.basename(args.prompt), content))
        print(f"Loaded prompt from file: {len(prompts[0][1])} chars")
    else:
        prompts.append(("prompt", args.prompt))

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"zimage-{timestamp}.png"

    # we handle output naming dynamically if multiple prompts or counts are involved
    base_output_dir = os.path.dirname(os.path.abspath(args.output))
    if not base_output_dir:
         base_output_dir = os.getcwd() # fallback

    if args.output and not os.path.isabs(args.output):
         args.output = os.path.abspath(args.output)

    base_output_name = os.path.basename(args.output)
    base_name_no_ext, ext = os.path.splitext(base_output_name)
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)


    # Model Config
    model_config = ModelConfig(
        name_or_path=args.transformer_path,
        is_flux=False,
        arch="zimage",
        dtype="bf16",
        layer_offloading=args.offload,
    )

    print("Initializing Model...")
    zimage_model = ZImageModel(
        device=args.device,
        model_config=model_config,
        dtype="bf16"
    )

    print("Loading Weights...")
    zimage_model.load_model()

    # Apply LoRA
    if args.lora and os.path.exists(args.lora):
        apply_lora(zimage_model.model, args.lora, scale=args.lora_scale)
    else:
        print(f"LoRA path {args.lora} does not exist!")
        return

    # Generate
    print("Generating...")

    gen_configs = []

    for prompt_filename, prompt_text in prompts:
        # print(f"Preparing configs for prompt: {prompt_filename}")

        for i in range(args.count):
            if args.seed is None:
                seed = -1
            else:
                seed = args.seed + i

            # Construct output path
            current_output_name = base_output_name

            if len(prompts) > 1:
                 p_name = os.path.splitext(prompt_filename)[0]
                 final_output_path = os.path.join(output_dir, f"{p_name}_{base_name_no_ext}{ext}")
            else:
                 final_output_path = os.path.join(output_dir, base_output_name)

            # verify [count] presence
            if (len(prompts) * args.count) > 1 and "[count]" not in final_output_path:
                 base, e = os.path.splitext(final_output_path)
                 final_output_path = f"{base}_[count]{e}"

            gen_config = GenerateImageConfig(
                prompt=prompt_text,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                output_path=final_output_path,
                seed=seed,
            )
            gen_configs.append(gen_config)

    print(f"Total image configs prepared: {len(gen_configs)}")
    if len(gen_configs) > 0:
        zimage_model.generate_images(gen_configs)
        print(f"Done.")
    else:
        print("No generation configs created. Check inputs.")

if __name__ == "__main__":
    main()
