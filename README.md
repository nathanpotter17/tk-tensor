# tk-tensors

A user-friendly Tkinter GUI for [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), enabling local AI image generation with FLUX, SDXL, SD3, and Qwen-Image models.

![TK TENSORS GUI](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)

## Features

- **Multi-Model Support**: SDXL, FLUX.1 (schnell/dev), SD3, Qwen-Image with quantized variants (Q4, Q5, Q8)
- **Intuitive Interface**: Clean, organized sidebar with real-time parameter controls
- **Live Preview**: Built-in image viewer with auto-scaling
- **Progress Tracking**: Real-time generation progress with detailed logging
- **Configuration Management**: Save and load generation settings as JSON
- **Smart Defaults**: Pre-configured optimal settings for each model type
- **Flexible Options**: CLIP on CPU, Flash Attention, VAE tiling, CPU offloading

## Requirements

### System Requirements
- **Python**: 3.7 or higher
- **RAM**: 8GB minimum (16GB+ recommended for larger models)
- **GPU**: NVIDIA CUDA or Apple Metal recommended (CPU fallback available)
- **Storage**: 5-20GB per model file

### Dependencies
```bash
pip install tkinter  # Usually included with Python
```

### External Requirements
- **sd-cli**: The stable-diffusion.cpp command-line executable
  - Build from source: [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
  - Place `sd-cli` (or `sd-cli.exe`) in the same directory as `sd_gui.py`
  - Place model files (`.safetensors`, `.gguf`) in the tk-tensors directory
    - For multi-file models (FLUX, Qwen), place all required files together

## Supported Models

### FLUX.1 Models (Recommended)
**FLUX.1-schnell** (4 steps, fast):
- `flux1-schnell-q4_0.gguf` (smallest)
- `flux1-schnell-q5_0.gguf` / `flux1-schnell-q5_1.gguf`
- `flux1-schnell-q8_0.gguf` (best quality)

**FLUX.1-dev** (20 steps, high quality):
- `flux1-dev-q4_0.gguf` / `flux1-dev-q4_k.gguf`
- `flux1-dev-q5_0.gguf` / `flux1-dev-q5_1.gguf` / `flux1-dev-q5_k.gguf`
- `flux1-dev-q8_0.gguf`

**Required files for FLUX**:
- `clip_l.safetensors`
- `t5xxl_fp16.safetensors`
- `ae.safetensors`

### Qwen-Image Models
- `qwen-image-2512-Q3_K_M.gguf` through `qwen-image-2512-Q8_0.gguf`

**Required files**:
- `Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf`
- `qwen_image_vae.safetensors`

### SDXL Models
- `sd_xl_base_1.0.safetensors`

## Usage

### Starting the GUI
```bash
python sd_gui.py
# or
python3 sd_gui.py
```

### Basic Workflow

1. **Select Model**: Choose from auto-detected models in the dropdown
2. **Set Resolution**: Pick a preset or enter custom dimensions
3. **Configure Parameters**:
   - **Steps**: 4 for FLUX-schnell, 20-40 for others
   - **CFG Scale**: 1.0-3.5 for FLUX, 5.0-12.0 for SD models
   - **Sampler**: euler for FLUX, euler_a for SD models
   - **Seed**: -1 for random, or a specific number for reproducibility

4. **Write Prompt**:
   - **FLUX**: Natural language works best (e.g., "A photorealistic image of...")
   - **SD/SDXL**: Comma-separated tags (e.g., "red fox, snow, 4k, detailed")

5. **Click Generate**: Progress appears in the Log tab, preview in Preview tab

### Example Prompts

**FLUX**:
```
A photorealistic image of a steampunk mechanical owl perched on an old 
leather-bound book in a dimly lit Victorian library. Warm candlelight 
illuminates intricate brass gears and copper feathers.
```

**SDXL**:
```
cinematic photo of an astronaut riding a horse through a neon city, 
cyberpunk, rain, reflections, volumetric lighting, 8k, photorealistic
```

## Configuration

### Save/Load Settings
- **Save Cfg**: Export current parameters to JSON
- **Load Cfg**: Import previously saved settings
- Saved files include: model, prompt, negative prompt, all parameters

### Options Explained

| Option | Description | When to Use |
|--------|-------------|-------------|
| **Keep CLIP on CPU** | Offload text encoder to CPU | Save VRAM, slight speed cost |
| **Flash Attention** | Faster attention (CUDA/Metal) | NVIDIA/Apple GPUs only |
| **VAE Tiling** | Process image in tiles | High resolution (>1536px) |
| **Offload to CPU** | Move models to RAM when idle | Low VRAM (<6GB) |

### Resolution Presets
- **512×512** - SD 1.5 native
- **1024×1024** - SDXL/FLUX standard
- **1328×1328** - High resolution
- **1536×1024** - Widescreen
- **Custom** - Enter any dimensions (divisible by 64)

## Tips & Best Practices

### Performance Optimization
- **Low VRAM (<8GB)**: Use Q4 models + "Offload to CPU" + "Keep CLIP on CPU"
- **Mid VRAM (8-16GB)**: Use Q5 models + "Keep CLIP on CPU"
- **High VRAM (16GB+)**: Use Q8 models, disable offloading

### Prompt Tips
- Be specific about subject, style, lighting, and mood
- Use negative prompts to avoid unwanted elements (SD/SDXL only)
- FLUX ignores negative prompts when CFG=1.0
- Longer, descriptive prompts work better for FLUX
- Comma-separated keywords work better for SD/SDXL

### Common CFG Scale Values
- **FLUX-schnell**: 1.0 (fixed)
- **FLUX-dev**: 2.5-3.5
- **SDXL**: 7.0-10.0
- **SD 1.5/2.x**: 7.0-12.0

## Troubleshooting

### "sd-cli not found"
- Build stable-diffusion.cpp and place the executable in the same folder
- Or click **Browse** to manually locate `sd-cli`

### "No models found"
- Download model files and place them in the same directory as `sd_gui.py`
- Ensure filenames match those in `ModelConfig.CONFIGS`

### CUDA/Metal errors
- Update GPU drivers
- Try disabling "Flash Attention"
- Enable "Offload to CPU" for VRAM issues

### Slow generation
- Use quantized models (Q4 faster than Q8)
- Enable "Keep CLIP on CPU"
- Enable "Flash Attention" if you have compatible GPU
- Reduce resolution or step count

### Generation fails/crashes
- Check the Log tab for error details
- Verify all required files are present (CLIP, VAE, etc.)
- Ensure enough disk space in `output/` directory
- Try lower resolution or enable VAE tiling

## Output

Generated images are saved to the `output/` directory with filenames:
```
img_YYYYMMDD_HHMMSS_s<seed>.png
```

## Project Structure

```
tk-tensors/
├── sd_gui.py              # Main GUI application
├── sd-cli(.exe)           # stable-diffusion.cpp executable - must obtain yourself
├── output/                # Generated images
├── *.safetensors          # Model weight files - must obtain yourself
├── *.gguf                 # Quantized model files - must obtain yourself
└── README.md              # This file
```

## Contributing

Contributions are welcome! Areas for improvement:
- Additional model support
- Advanced features (LoRA, ControlNet, img2img)
- UI/UX enhancements
- Performance optimizations

## Acknowledgments

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) - The underlying inference engine
- FLUX, SDXL, and Qwen-Image model creators
- The open-source AI community

## License

This project is provided as-is for educational and research purposes. Please respect the licenses of the underlying models and stable-diffusion.cpp.