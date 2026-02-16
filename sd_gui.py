#!/usr/bin/env python3

import subprocess, sys, os, json, threading, queue, time, signal, re, shutil, math
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# Model Configuration
# ═══════════════════════════════════════════════════════════════

class ModelType(Enum):
    SD15       = "sd1.5"
    SD20       = "sd2.x"
    SDXL       = "sdxl"
    SD3        = "sd3"
    FLUX       = "flux"
    FLUX2      = "flux2"
    QWEN_IMAGE = "qwen_image"

class ModelConfig:
    """
    Registry of known image model files.
    Each entry maps a filename (or diffusion-model filename for multi-file models)
    to its metadata and default generation parameters.
    """
    CONFIGS = {
        # ── SDXL ────────────────────────────────────────────
        "sd_xl_base_1.0.safetensors": {
            "type": ModelType.SDXL,
            "name": "Stable Diffusion XL 1.0",
            "default_steps": "25", "default_cfg": "7.0",
            "default_width": "1024", "default_height": "1024",
            "default_sampler": "euler_a", "default_seed": "-1",
            "model_flag": "-m", "extra_files": {},
            "clip_on_cpu": False, "flash_attn": False,
        },
        # ── FLUX.1-schnell (GGUF quantized) ────────────────
        "flux1-schnell-q4_0.gguf": {
            "type": ModelType.FLUX,
            "name": "FLUX.1 Schnell (Q4)",
            "default_steps": "4", "default_cfg": "1.0",
            "default_width": "1024", "default_height": "1024",
            "default_sampler": "euler", "default_seed": "-1",
            "model_flag": "--diffusion-model",
            "extra_files": {
                "--clip_l": "clip_l.safetensors",
                "--t5xxl":  "t5xxl_fp16.safetensors",
                "--vae":    "ae.safetensors",
            },
            "clip_on_cpu": True, "flash_attn": True,
        },
        "flux1-schnell-q5_0.gguf": {
            "type": ModelType.FLUX,
            "name": "FLUX.1 Schnell (Q5)",
            "default_steps": "4", "default_cfg": "1.0",
            "default_width": "1024", "default_height": "1024",
            "default_sampler": "euler", "default_seed": "-1",
            "model_flag": "--diffusion-model",
            "extra_files": {
                "--clip_l": "clip_l.safetensors",
                "--t5xxl":  "t5xxl_fp16.safetensors",
                "--vae":    "ae.safetensors",
            },
            "clip_on_cpu": True, "flash_attn": True,
        },
        "flux1-schnell-q5_1.gguf": {
            "type": ModelType.FLUX,
            "name": "FLUX.1 Schnell (Q5_1)",
            "default_steps": "4", "default_cfg": "1.0",
            "default_width": "1024", "default_height": "1024",
            "default_sampler": "euler", "default_seed": "-1",
            "model_flag": "--diffusion-model",
            "extra_files": {
                "--clip_l": "clip_l.safetensors",
                "--t5xxl":  "t5xxl_fp16.safetensors",
                "--vae":    "ae.safetensors",
            },
            "clip_on_cpu": True, "flash_attn": True,
        },
        "flux1-schnell-q8_0.gguf": {
            "type": ModelType.FLUX,
            "name": "FLUX.1 Schnell (Q8)",
            "default_steps": "4", "default_cfg": "1.0",
            "default_width": "1024", "default_height": "1024",
            "default_sampler": "euler", "default_seed": "-1",
            "model_flag": "--diffusion-model",
            "extra_files": {
                "--clip_l": "clip_l.safetensors",
                "--t5xxl":  "t5xxl_fp16.safetensors",
                "--vae":    "ae.safetensors",
            },
            "clip_on_cpu": True, "flash_attn": True,
        },
        # ── FLUX.1-dev (GGUF quantized) ────────────────────
        "flux1-dev-q4_0.gguf": {
            "type": ModelType.FLUX,
            "name": "FLUX.1 Dev (Q4)",
            "default_steps": "20", "default_cfg": "3.5",
            "default_width": "1024", "default_height": "1024",
            "default_sampler": "euler", "default_seed": "-1",
            "model_flag": "--diffusion-model",
            "extra_files": {
                "--clip_l": "clip_l.safetensors",
                "--t5xxl":  "t5xxl_fp16.safetensors",
                "--vae":    "ae.safetensors",
            },
            "clip_on_cpu": True, "flash_attn": True,
        },
        "flux1-dev-q4_k.gguf": {
            "type": ModelType.FLUX,
            "name": "FLUX.1 Dev (Q4_K)",
            "default_steps": "20", "default_cfg": "3.5",
            "default_width": "1024", "default_height": "1024",
            "default_sampler": "euler", "default_seed": "-1",
            "model_flag": "--diffusion-model",
            "extra_files": {
                "--clip_l": "clip_l.safetensors",
                "--t5xxl":  "t5xxl_fp16.safetensors",
                "--vae":    "ae.safetensors",
            },
            "clip_on_cpu": True, "flash_attn": True,
        },
        "flux1-dev-q5_0.gguf": {
            "type": ModelType.FLUX,
            "name": "FLUX.1 Dev (Q5)",
            "default_steps": "20", "default_cfg": "3.5",
            "default_width": "1024", "default_height": "1024",
            "default_sampler": "euler", "default_seed": "-1",
            "model_flag": "--diffusion-model",
            "extra_files": {
                "--clip_l": "clip_l.safetensors",
                "--t5xxl":  "t5xxl_fp16.safetensors",
                "--vae":    "ae.safetensors",
            },
            "clip_on_cpu": True, "flash_attn": True,
        },
        "flux1-dev-q5_1.gguf": {
            "type": ModelType.FLUX,
            "name": "FLUX.1 Dev (Q5_1)",
            "default_steps": "20", "default_cfg": "3.5",
            "default_width": "1024", "default_height": "1024",
            "default_sampler": "euler", "default_seed": "-1",
            "model_flag": "--diffusion-model",
            "extra_files": {
                "--clip_l": "clip_l.safetensors",
                "--t5xxl":  "t5xxl_fp16.safetensors",
                "--vae":    "ae.safetensors",
            },
            "clip_on_cpu": True, "flash_attn": True,
        },
        "flux1-dev-q5_k.gguf": {
            "type": ModelType.FLUX,
            "name": "FLUX.1 Dev (Q5_K)",
            "default_steps": "20", "default_cfg": "3.5",
            "default_width": "1024", "default_height": "1024",
            "default_sampler": "euler", "default_seed": "-1",
            "model_flag": "--diffusion-model",
            "extra_files": {
                "--clip_l": "clip_l.safetensors",
                "--t5xxl":  "t5xxl_fp16.safetensors",
                "--vae":    "ae.safetensors",
            },
            "clip_on_cpu": True, "flash_attn": True,
        },
        "flux1-dev-q8_0.gguf": {
            "type": ModelType.FLUX,
            "name": "FLUX.1 Dev (Q8)",
            "default_steps": "20", "default_cfg": "3.5",
            "default_width": "1024", "default_height": "1024",
            "default_sampler": "euler", "default_seed": "-1",
            "model_flag": "--diffusion-model",
            "extra_files": {
                "--clip_l": "clip_l.safetensors",
                "--t5xxl":  "t5xxl_fp16.safetensors",
                "--vae":    "ae.safetensors",
            },
            "clip_on_cpu": True, "flash_attn": True,
        },
        # ── Qwen-Image-2512 (GGUF quantized) ──────────────
        "qwen-image-2512-Q3_K_M.gguf": {
            "type": ModelType.QWEN_IMAGE,
            "name": "Qwen-Image 2512 (Q3_K_M)",
            "default_steps": "40", "default_cfg": "2.5",
            "default_width": "1024", "default_height": "1024",
            "default_sampler": "euler", "default_seed": "-1",
            "model_flag": "--diffusion-model",
            "extra_files": {
                "--llm": "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
                "--vae": "qwen_image_vae.safetensors",
            },
            "clip_on_cpu": False, "flash_attn": True,
            "flow_shift": "3", "offload_to_cpu": True,
        },
        "qwen-image-2512-Q4_K_M.gguf": {
            "type": ModelType.QWEN_IMAGE,
            "name": "Qwen-Image 2512 (Q4_K_M)",
            "default_steps": "40", "default_cfg": "2.5",
            "default_width": "1024", "default_height": "1024",
            "default_sampler": "euler", "default_seed": "-1",
            "model_flag": "--diffusion-model",
            "extra_files": {
                "--llm": "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
                "--vae": "qwen_image_vae.safetensors",
            },
            "clip_on_cpu": False, "flash_attn": True,
            "flow_shift": "3", "offload_to_cpu": True,
        },
        "qwen-image-2512-Q5_K_M.gguf": {
            "type": ModelType.QWEN_IMAGE,
            "name": "Qwen-Image 2512 (Q5_K_M)",
            "default_steps": "40", "default_cfg": "2.5",
            "default_width": "1024", "default_height": "1024",
            "default_sampler": "euler", "default_seed": "-1",
            "model_flag": "--diffusion-model",
            "extra_files": {
                "--llm": "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
                "--vae": "qwen_image_vae.safetensors",
            },
            "clip_on_cpu": False, "flash_attn": True,
            "flow_shift": "3", "offload_to_cpu": True,
        },
        "qwen-image-2512-Q6_K.gguf": {
            "type": ModelType.QWEN_IMAGE,
            "name": "Qwen-Image 2512 (Q6_K)",
            "default_steps": "40", "default_cfg": "2.5",
            "default_width": "1024", "default_height": "1024",
            "default_sampler": "euler", "default_seed": "-1",
            "model_flag": "--diffusion-model",
            "extra_files": {
                "--llm": "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
                "--vae": "qwen_image_vae.safetensors",
            },
            "clip_on_cpu": False, "flash_attn": True,
            "flow_shift": "3", "offload_to_cpu": True,
        },
        "qwen-image-2512-Q8_0.gguf": {
            "type": ModelType.QWEN_IMAGE,
            "name": "Qwen-Image 2512 (Q8)",
            "default_steps": "40", "default_cfg": "2.5",
            "default_width": "1024", "default_height": "1024",
            "default_sampler": "euler", "default_seed": "-1",
            "model_flag": "--diffusion-model",
            "extra_files": {
                "--llm": "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
                "--vae": "qwen_image_vae.safetensors",
            },
            "clip_on_cpu": False, "flash_attn": True,
            "flow_shift": "3", "offload_to_cpu": True,
        },
    }

    SAMPLERS = [
        "euler_a", "euler", "heun", "dpm2", "dpm++2s_a", "dpm++2m",
        "dpm++2mv2", "ipndm", "ipndm_v", "lcm",
    ]

    RESOLUTIONS = [
        ("512 x 512",   512,  512),
        ("512 x 768",   512,  768),
        ("768 x 512",   768,  512),
        ("768 x 768",   768,  768),
        ("1024 x 1024", 1024, 1024),
        ("1024 x 768",  1024, 768),
        ("768 x 1024",  768,  1024),
        ("1024 x 576",  1024, 576),
        ("576 x 1024",  576,  1024),
        ("1280 x 720",  1280, 720),
        ("1328 x 1328", 1328, 1328),
        ("1536 x 1024", 1536, 1024),
        ("Custom", 0, 0),
    ]


# ═══════════════════════════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════════════════════════

class SDGui:
    SIDEBAR_W = 290

    def __init__(self, root):
        self.root = root
        self.root.title("TK TENSORS")
        self.root.geometry("1200x800")
        self.root.minsize(900, 560)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Sidebar.TLabelframe",        padding=(6, 4))
        style.configure("Sidebar.TLabelframe.Label",   font=("", 9, "bold"))
        style.configure("Sidebar.TLabel",               font=("", 9))
        style.configure("SidebarSmall.TLabel",          font=("", 8), foreground="gray")

        self.sd_cli = self._find_sd_cli()
        self.selected_model = tk.StringVar()
        self.output_dir = os.path.abspath("output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.process = None
        self.output_queue = queue.Queue()
        self.generation_thread = None
        self.is_generating = False
        self.last_image_path = None
        self._preview_photo = None
        self._preview_full = None

        self.available_models = self._detect_models()

        self._build_ui()

        if self.available_models:
            self.selected_model.set(self.available_models[0])
            self._on_model_change()
        self._process_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    # ── Helpers ────────────────────────────────────────────────
    @staticmethod
    def _find_sd_cli():
        candidates = [
            "sd-cli.exe", "./sd-cli.exe", "sd-cli", "./sd-cli",
            "../sd-cli.exe", "../sd-cli",
            "bin/sd-cli.exe", "bin/sd-cli",
            "build/bin/sd-cli.exe", "build/bin/sd-cli",
            "../stable-diffusion.cpp/build/bin/sd-cli.exe",
            "../stable-diffusion.cpp/build/bin/sd-cli",
        ]
        for p in candidates:
            if os.path.isfile(p):
                return os.path.abspath(p)
        messagebox.showwarning(
            "sd-cli not found",
            "Could not locate sd-cli executable.\n"
            "Please build stable-diffusion.cpp and place sd-cli\n"
            "in this directory, or use Browse to locate it."
        )
        return "sd-cli.exe" if sys.platform == "win32" else "sd-cli"

    @staticmethod
    def _detect_models():
        found = []
        for filename in ModelConfig.CONFIGS:
            if Path(filename).exists():
                found.append(filename)
        return found

    def _on_closing(self):
        if self.is_generating and not messagebox.askokcancel(
                "Quit", "Generation is in progress. Force quit?"):
            return
        if self.is_generating:
            self._force_shutdown()
        else:
            self.root.destroy()

    # ── Top-level layout ──────────────────────────────────────
    def _build_ui(self):
        # ── Toolbar ──
        toolbar = ttk.Frame(self.root, padding="4 4 4 4")
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.generate_btn = ttk.Button(
            toolbar, text="Generate", command=self._generate,
            state=tk.NORMAL if self.available_models else tk.DISABLED)
        self.generate_btn.pack(side=tk.LEFT, padx=2)

        self.stop_btn = ttk.Button(
            toolbar, text="Stop", command=self._stop_generation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        self.kill_btn = ttk.Button(
            toolbar, text="KILL", command=self._force_shutdown, state=tk.DISABLED)
        self.kill_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(toolbar, text="Save Cfg",  command=self._save_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Load Cfg",  command=self._load_config).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(toolbar, text="Open Output Dir", command=self._open_output_dir).pack(side=tk.LEFT, padx=2)

        self.model_info_label = ttk.Label(
            toolbar, text="", foreground="#2266aa", font=("", 9, "bold"))
        self.model_info_label.pack(side=tk.RIGHT, padx=8)

        ttk.Separator(self.root).pack(side=tk.TOP, fill=tk.X)

        # ── Status bar ──
        sb = ttk.Frame(self.root)
        sb.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(sb, textvariable=self.status_var, relief=tk.SUNKEN,
                  anchor="w").pack(fill=tk.X, padx=4, pady=2)

        # ── Progress bar ──
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            sb, variable=self.progress_var, maximum=100, mode="determinate")
        self.progress_bar.pack(fill=tk.X, padx=4, pady=(0, 2))

        # ── Body ──
        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True)

        self._build_sidebar(body)

        self.notebook = ttk.Notebook(body)
        self.notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 5), pady=5)

        prompt_tab = ttk.Frame(self.notebook)
        self.notebook.add(prompt_tab, text="  Prompt  ")
        self._build_prompt_tab(prompt_tab)

        preview_tab = ttk.Frame(self.notebook)
        self.notebook.add(preview_tab, text="  Preview  ")
        self._build_preview_tab(preview_tab)

        log_tab = ttk.Frame(self.notebook)
        self.notebook.add(log_tab, text="  Log  ")
        self._build_log_tab(log_tab)

    # ── Sidebar ───────────────────────────────────────────────
    def _build_sidebar(self, parent):
        outer = ttk.Frame(parent, width=self.SIDEBAR_W)
        outer.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0), pady=5)
        outer.pack_propagate(False)

        # ── Model ──
        mf = ttk.LabelFrame(outer, text="  Model  ", style="Sidebar.TLabelframe")
        mf.pack(fill=tk.X, pady=(0, 4))

        if self.available_models:
            cb = ttk.Combobox(
                mf, textvariable=self.selected_model,
                values=self.available_models, state="readonly",
                font=("", 9), width=32)
            cb.pack(fill=tk.X, padx=4, pady=(2, 4))
            cb.bind("<<ComboboxSelected>>", lambda e: self._on_model_change())
        else:
            ttk.Label(
                mf, text="No models found in working dir.\n"
                "See BUILD_INSTRUCTIONS.md for download links.",
                foreground="red", wraplength=self.SIDEBAR_W - 30,
                style="Sidebar.TLabel").pack(padx=4, pady=4)

        exe_row = ttk.Frame(mf)
        exe_row.pack(fill=tk.X, padx=4, pady=(0, 4))
        self.exec_label = ttk.Label(
            exe_row, text=os.path.basename(self.sd_cli),
            style="SidebarSmall.TLabel", anchor="w")
        self.exec_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(exe_row, text="Browse", width=7,
                   command=self._browse_executable).pack(side=tk.RIGHT, padx=(4, 0))

        # ── Resolution ──
        rf = ttk.LabelFrame(outer, text="  Resolution  ", style="Sidebar.TLabelframe")
        rf.pack(fill=tk.X, pady=(0, 4))

        rgrid = ttk.Frame(rf)
        rgrid.pack(fill=tk.X, padx=4, pady=4)
        rgrid.columnconfigure(1, weight=1)

        self.resolution_var = tk.StringVar(value="1024 x 1024")
        ttk.Label(rgrid, text="Preset", style="Sidebar.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=2)
        res_combo = ttk.Combobox(
            rgrid, textvariable=self.resolution_var,
            values=[r[0] for r in ModelConfig.RESOLUTIONS],
            state="readonly", font=("", 9), width=14)
        res_combo.grid(row=0, column=1, sticky="ew", pady=2)
        res_combo.bind("<<ComboboxSelected>>", lambda e: self._on_resolution_change())

        self.params = {}

        ttk.Label(rgrid, text="Width", style="Sidebar.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=2)
        w_entry = ttk.Entry(rgrid, width=10, font=("", 9))
        w_entry.insert(0, "1024")
        w_entry.grid(row=1, column=1, sticky="ew", pady=2)
        self.params["width"] = w_entry

        ttk.Label(rgrid, text="Height", style="Sidebar.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=2)
        h_entry = ttk.Entry(rgrid, width=10, font=("", 9))
        h_entry.insert(0, "1024")
        h_entry.grid(row=2, column=1, sticky="ew", pady=2)
        self.params["height"] = h_entry

        # ── Parameters ──
        pf = ttk.LabelFrame(outer, text="  Parameters  ", style="Sidebar.TLabelframe")
        pf.pack(fill=tk.X, pady=(0, 4))

        pgrid = ttk.Frame(pf)
        pgrid.pack(fill=tk.X, padx=4, pady=4)
        pgrid.columnconfigure(1, weight=1)

        param_defs = [
            ("Steps",      "steps",      "20"),
            ("CFG Scale",  "cfg_scale",  "7.5"),
            ("Seed",       "seed",       "-1"),
            ("Flow Shift", "flow_shift", "0"),
            ("Threads",    "threads",    "8"),
            ("Batch",      "batch",      "1"),
        ]
        for i, (label, key, default) in enumerate(param_defs):
            ttk.Label(pgrid, text=label, style="Sidebar.TLabel").grid(
                row=i, column=0, sticky="w", padx=(0, 8), pady=2)
            e = ttk.Entry(pgrid, width=10, font=("", 9))
            e.insert(0, default)
            e.grid(row=i, column=1, sticky="ew", pady=2)
            self.params[key] = e

        ttk.Label(pgrid, text="Sampler", style="Sidebar.TLabel").grid(
            row=len(param_defs), column=0, sticky="w", padx=(0, 8), pady=2)
        self.sampler_var = tk.StringVar(value="euler_a")
        ttk.Combobox(
            pgrid, textvariable=self.sampler_var,
            values=ModelConfig.SAMPLERS, state="readonly",
            font=("", 9), width=14
        ).grid(row=len(param_defs), column=1, sticky="ew", pady=2)

        # ── Options ──
        of = ttk.LabelFrame(outer, text="  Options  ", style="Sidebar.TLabelframe")
        of.pack(fill=tk.X, pady=(0, 4))

        self.clip_cpu_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(of, text="Keep CLIP on CPU (save VRAM)",
                        variable=self.clip_cpu_var).pack(anchor="w", padx=4, pady=2)

        self.flash_attn_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(of, text="Flash Attention (CUDA/Metal)",
                        variable=self.flash_attn_var).pack(anchor="w", padx=4, pady=2)

        self.vae_tiling_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(of, text="VAE Tiling (high-res, save VRAM)",
                        variable=self.vae_tiling_var).pack(anchor="w", padx=4, pady=2)

        self.offload_cpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(of, text="Offload to CPU (low VRAM)",
                        variable=self.offload_cpu_var).pack(anchor="w", padx=4, pady=2)

    # ── Prompt tab ────────────────────────────────────────────
    def _build_prompt_tab(self, parent):
        f = ttk.Frame(parent, padding="8")
        f.pack(fill=tk.BOTH, expand=True)

        btn = ttk.Frame(f)
        btn.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(btn, text="Load File",  command=self._load_prompt_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn, text="Example",    command=self._load_example).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn, text="Clear",      command=self._clear_prompts).pack(side=tk.LEFT, padx=2)

        ttk.Label(f, text="Prompt:").pack(anchor="w")
        self.prompt_text = scrolledtext.ScrolledText(f, height=6, wrap=tk.WORD, font=("", 10))
        self.prompt_text.pack(fill=tk.X, pady=(2, 8))

        ttk.Label(f, text="Negative prompt (what to avoid):").pack(anchor="w")
        self.neg_prompt_text = scrolledtext.ScrolledText(f, height=4, wrap=tk.WORD, font=("", 10))
        self.neg_prompt_text.pack(fill=tk.X, pady=(2, 8))
        self.neg_prompt_text.insert(
            1.0, "ugly, blurry, low quality, deformed, disfigured, watermark, text")

        info_frame = ttk.LabelFrame(f, text="  Quick Reference  ")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        info_text = (
            "Prompt tips:\n"
            "- Be descriptive: subject, style, lighting, camera, mood\n"
            "- Example: \"a red fox in a snowy forest, digital painting, "
            "dramatic lighting, 4k, detailed\"\n"
            "- FLUX models: use natural language, longer prompts work well\n"
            "- SD models: comma-separated tags work best\n"
            "- CFG scale: higher = more prompt adherence, lower = more creative\n"
            "  (FLUX uses cfg 1.0-3.5, SD uses 5.0-12.0)\n"
            "- Steps: FLUX-schnell needs only 4, SD models need 20-30\n"
            "- Negative prompt is ignored by FLUX (cfg_scale=1.0 disables it)"
        )
        ttk.Label(info_frame, text=info_text, style="Sidebar.TLabel",
                  wraplength=600, justify="left").pack(padx=8, pady=8, anchor="w")

    # ── Preview tab ───────────────────────────────────────────
    def _build_preview_tab(self, parent):
        f = ttk.Frame(parent, padding="8")
        f.pack(fill=tk.BOTH, expand=True)

        btn = ttk.Frame(f)
        btn.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(btn, text="Save As",        command=self._save_image_as).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn, text="Open in Viewer",  command=self._open_in_viewer).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn, text="Copy Path",       command=self._copy_image_path).pack(side=tk.LEFT, padx=2)

        self.preview_info_var = tk.StringVar(value="No image generated yet.")
        ttk.Label(f, textvariable=self.preview_info_var,
                  foreground="gray").pack(anchor="w", pady=(0, 4))

        self.preview_canvas = tk.Canvas(f, bg="#1e1e1e", highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas.bind("<Configure>", self._on_canvas_resize)

    # ── Log tab ───────────────────────────────────────────────
    def _build_log_tab(self, parent):
        f = ttk.Frame(parent, padding="8")
        f.pack(fill=tk.BOTH, expand=True)

        btn = ttk.Frame(f)
        btn.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(btn, text="Clear", command=self._clear_log).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn, text="Copy",  command=self._copy_log).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn, text="Save",  command=self._save_log).pack(side=tk.LEFT, padx=2)
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(btn, text="Auto-scroll",
                        variable=self.auto_scroll_var).pack(side=tk.LEFT, padx=12)

        self.log_text = scrolledtext.ScrolledText(
            f, wrap=tk.WORD, bg="#1e1e1e", fg="#d4d4d4",
            font=("Consolas", 10) if sys.platform == "win32" else ("Courier", 10),
            insertbackground="white", selectbackground="#264f78")
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
        for tag, color in [
            ("info",    "#dcdcaa"), ("error",   "#f44747"),
            ("success", "#6a9955"), ("warning", "#ce9178"),
            ("step",    "#569cd6"),
        ]:
            self.log_text.tag_config(tag, foreground=color)

    # ── Sidebar callbacks ─────────────────────────────────────
    def _browse_executable(self):
        fp = filedialog.askopenfilename(
            title="Select sd-cli executable",
            filetypes=[("Executable",
                        "*.exe" if sys.platform == "win32" else "*"),
                       ("All", "*.*")])
        if fp and os.path.isfile(fp):
            self.sd_cli = os.path.abspath(fp)
            self.exec_label.config(text=os.path.basename(self.sd_cli))
            self.status_var.set(f"Executable: {os.path.basename(self.sd_cli)}")

    def _on_model_change(self):
        model = self.selected_model.get()
        if model not in ModelConfig.CONFIGS:
            return
        cfg = ModelConfig.CONFIGS[model]
        self.model_info_label.config(text=cfg["name"])

        mapping = {
            "steps":     "default_steps",
            "cfg_scale": "default_cfg",
            "width":     "default_width",
            "height":    "default_height",
            "seed":      "default_seed",
        }
        for pk, ck in mapping.items():
            w = self.params[pk]
            w.delete(0, tk.END)
            w.insert(0, str(cfg[ck]))

        self.sampler_var.set(cfg["default_sampler"])
        self.clip_cpu_var.set(cfg["clip_on_cpu"])
        self.flash_attn_var.set(cfg["flash_attn"])
        self.offload_cpu_var.set(cfg.get("offload_to_cpu", False))

        fs = cfg.get("flow_shift", "0")
        self.params["flow_shift"].delete(0, tk.END)
        self.params["flow_shift"].insert(0, fs)

        w, h = int(cfg["default_width"]), int(cfg["default_height"])
        matched = False
        for label, rw, rh in ModelConfig.RESOLUTIONS:
            if rw == w and rh == h:
                self.resolution_var.set(label)
                matched = True
                break
        if not matched:
            self.resolution_var.set("Custom")

        missing = []
        for flag, fname in cfg.get("extra_files", {}).items():
            if not Path(fname).exists():
                missing.append(fname)
        if missing:
            self.status_var.set(f"Warning: missing files: {', '.join(missing)}")

    def _on_resolution_change(self):
        label = self.resolution_var.get()
        for res_label, w, h in ModelConfig.RESOLUTIONS:
            if res_label == label and w > 0:
                self.params["width"].delete(0, tk.END)
                self.params["width"].insert(0, str(w))
                self.params["height"].delete(0, tk.END)
                self.params["height"].insert(0, str(h))
                break

    # ── Prompt helpers ────────────────────────────────────────
    def _load_prompt_file(self):
        fp = filedialog.askopenfilename(
            title="Select prompt file",
            filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if fp:
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    self.prompt_text.delete(1.0, tk.END)
                    self.prompt_text.insert(1.0, fh.read())
                    self.status_var.set(f"Loaded: {Path(fp).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")

    def _clear_prompts(self):
        self.prompt_text.delete(1.0, tk.END)
        self.neg_prompt_text.delete(1.0, tk.END)
        self.status_var.set("Cleared")

    def _clear_log(self):
        self.log_text.delete(1.0, tk.END)

    def _copy_log(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.log_text.get(1.0, tk.END))
        self.status_var.set("Log copied to clipboard")

    def _save_log(self):
        fp = filedialog.asksaveasfilename(
            title="Save log", defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if fp:
            try:
                with open(fp, "w", encoding="utf-8") as fh:
                    fh.write(self.log_text.get(1.0, tk.END))
                self.status_var.set(f"Saved: {Path(fp).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {e}")

    def _load_example(self):
        model = self.selected_model.get()
        if model not in ModelConfig.CONFIGS:
            return
        mtype = ModelConfig.CONFIGS[model]["type"]

        examples = {
            ModelType.SD15: (
                "a beautiful landscape painting of mountains at sunset, "
                "oil on canvas, detailed, vibrant colors, golden hour lighting",
                "ugly, blurry, low quality, deformed, watermark"
            ),
            ModelType.SDXL: (
                "cinematic photo of an astronaut riding a horse through a neon city, "
                "cyberpunk, rain, reflections, volumetric lighting, 8k, photorealistic",
                "ugly, blurry, low quality, cartoon, painting, illustration"
            ),
            ModelType.SD3: (
                "A close-up photograph of a red fox in a snowy forest clearing. "
                "Soft morning light filters through the trees. The fox looks directly "
                "at the camera with bright amber eyes. Shot on Hasselblad, 85mm f/1.4.",
                "ugly, blurry, deformed, low quality, watermark"
            ),
            ModelType.FLUX: (
                "A photorealistic image of a steampunk mechanical owl perched on an "
                "old leather-bound book in a dimly lit Victorian library. Warm candlelight "
                "illuminates intricate brass gears and copper feathers. Dust motes float "
                "in the air. Shot with a macro lens, shallow depth of field.",
                ""
            ),
            ModelType.QWEN_IMAGE: (
                "A photograph of a Japanese garden in autumn with a stone bridge "
                "over a koi pond. Red maple leaves float on the still water surface, "
                "reflecting the golden afternoon light. A weathered wooden sign reads "
                "\"Peaceful Garden\" in elegant calligraphy. Shot on medium format film, "
                "soft natural light, 8K resolution.",
                ""
            ),
        }
        default = (
            "a lovely cat sitting on a windowsill, sunlight, cozy, detailed",
            "ugly, blurry, low quality"
        )
        prompt, neg = examples.get(mtype, default)
        self.prompt_text.delete(1.0, tk.END)
        self.prompt_text.insert(1.0, prompt)
        self.neg_prompt_text.delete(1.0, tk.END)
        self.neg_prompt_text.insert(1.0, neg)

    # ── Preview / image display ───────────────────────────────
    def _show_preview(self, image_path):
        """Display generated PNG in the preview canvas using tk.PhotoImage."""
        self.last_image_path = image_path
        try:
            full = tk.PhotoImage(file=image_path)
            iw, ih = full.width(), full.height()

            cw = max(self.preview_canvas.winfo_width(), 200)
            ch = max(self.preview_canvas.winfo_height(), 200)

            factor = max(1, math.ceil(iw / cw), math.ceil(ih / ch))

            if factor > 1:
                scaled = full.subsample(factor, factor)
            else:
                scaled = full

            self._preview_photo = scaled
            self._preview_full = full

            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                cw // 2, ch // 2, anchor=tk.CENTER, image=scaled)

            sw, sh = scaled.width(), scaled.height()
            self.preview_info_var.set(
                f"{os.path.basename(image_path)}  --  {iw}x{ih}"
                f"  (preview {sw}x{sh} at 1/{factor})")

        except Exception as e:
            self.preview_info_var.set(f"Preview error: {e}")
            self._preview_photo = None
            self._preview_full = None

    def _on_canvas_resize(self, event=None):
        if self.last_image_path and os.path.isfile(self.last_image_path):
            self._show_preview(self.last_image_path)

    def _open_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(self.output_dir)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", self.output_dir])
        else:
            subprocess.Popen(["xdg-open", self.output_dir])

    def _save_image_as(self):
        if not self.last_image_path or not os.path.isfile(self.last_image_path):
            messagebox.showinfo("No image", "No image generated yet.")
            return
        fp = filedialog.asksaveasfilename(
            title="Save image as",
            initialfile=os.path.basename(self.last_image_path),
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if fp:
            shutil.copy2(self.last_image_path, fp)
            self.status_var.set(f"Saved: {Path(fp).name}")

    def _open_in_viewer(self):
        if not self.last_image_path or not os.path.isfile(self.last_image_path):
            messagebox.showinfo("No image", "No image generated yet.")
            return
        if sys.platform == "win32":
            os.startfile(self.last_image_path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", self.last_image_path])
        else:
            subprocess.Popen(["xdg-open", self.last_image_path])

    def _copy_image_path(self):
        if self.last_image_path:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.last_image_path)
            self.status_var.set("Path copied to clipboard")

    # ── Generation ────────────────────────────────────────────
    def _generate(self):
        if not self.available_models:
            messagebox.showerror("Error", "No models available!")
            return
        if self.is_generating:
            messagebox.showwarning("Warning", "Already generating!")
            return

        prompt = self.prompt_text.get(1.0, tk.END).strip()
        if not prompt:
            messagebox.showwarning("Warning", "Enter a prompt!")
            return

        neg_prompt = self.neg_prompt_text.get(1.0, tk.END).strip()
        model_file = self.selected_model.get()
        cfg = ModelConfig.CONFIGS[model_file]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed = self.params["seed"].get().strip()
        output_path = os.path.join(self.output_dir, f"img_{ts}_s{seed}.png")

        cmd = [self.sd_cli]
        cmd.extend([cfg["model_flag"], model_file])

        for flag, fname in cfg.get("extra_files", {}).items():
            cmd.extend([flag, fname])

        cmd.extend(["-p", prompt])
        if neg_prompt:
            cmd.extend(["-n", neg_prompt])

        cmd.extend([
            "--steps",            self.params["steps"].get().strip(),
            "--cfg-scale",        self.params["cfg_scale"].get().strip(),
            "-W",                 self.params["width"].get().strip(),
            "-H",                 self.params["height"].get().strip(),
            "-s",                 self.params["seed"].get().strip(),
            "--sampling-method",  self.sampler_var.get(),
            "-t",                 self.params["threads"].get().strip(),
            "-o",                 output_path,
        ])

        batch = self.params["batch"].get().strip()
        if batch and int(batch) > 1:
            cmd.extend(["-b", batch])

        if self.clip_cpu_var.get():
            cmd.append("--clip-on-cpu")
        if self.flash_attn_var.get():
            cmd.append("--diffusion-fa")
        if self.vae_tiling_var.get():
            cmd.append("--vae-tiling")
        if self.offload_cpu_var.get():
            cmd.append("--offload-to-cpu")

        flow_shift = self.params["flow_shift"].get().strip()
        if flow_shift and flow_shift != "0":
            cmd.extend(["--flow-shift", flow_shift])

        # Switch to log and display header
        self.notebook.select(2)
        self._clear_log()
        sep = "=" * 60
        self.log_text.insert(tk.END, f"{sep}\n", "info")
        self.log_text.insert(tk.END, f"  {cfg['name']}\n", "info")
        self.log_text.insert(tk.END,
            f"  {self.params['width'].get()}x{self.params['height'].get()}"
            f"  |  {self.params['steps'].get()} steps"
            f"  |  cfg {self.params['cfg_scale'].get()}"
            f"  |  {self.sampler_var.get()}\n", "info")
        self.log_text.insert(tk.END, f"{sep}\n", "info")
        self.log_text.insert(tk.END, f"Output: {output_path}\n\n", "info")

        safe_cmd = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        self.log_text.insert(tk.END, f"$ {safe_cmd}\n\n", "warning")

        self._set_gen(True)
        self.progress_var.set(0)
        self.status_var.set("Generating...")

        self.generation_thread = threading.Thread(
            target=self._gen_worker, args=(cmd, output_path), daemon=True)
        self.generation_thread.start()

    def _set_gen(self, on):
        self.is_generating = on
        self.generate_btn.config(state=tk.DISABLED if on else tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL if on else tk.DISABLED)
        self.kill_btn.config(state=tk.NORMAL if on else tk.DISABLED)

    def _gen_worker(self, cmd, output_path):
        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1,
                encoding="utf-8", errors="replace")

            for line in self.process.stdout:
                self.output_queue.put(("output", line))

                m = re.search(r"step\s+(\d+)\s*/\s*(\d+)", line, re.IGNORECASE)
                if m:
                    current, total = int(m.group(1)), int(m.group(2))
                    pct = current / total * 100 if total > 0 else 0
                    self.output_queue.put(("progress", pct))

            self.process.wait()
            rc = self.process.returncode

            if rc == 0 and os.path.isfile(output_path):
                self.output_queue.put(("done", output_path))
            else:
                self.output_queue.put(("error",
                    f"sd-cli exited with code {rc}. Check log for details.\n"))
                self.output_queue.put(("done", None))

        except Exception as e:
            self.output_queue.put(("error", f"Error: {e}\n"))
            self.output_queue.put(("done", None))

    def _process_queue(self):
        try:
            while not self.output_queue.empty():
                tag, data = self.output_queue.get_nowait()

                if tag == "output":
                    line_tag = None
                    if "[ERROR]" in data or "error" in data.lower():
                        line_tag = "error"
                    elif "[WARN" in data:
                        line_tag = "warning"
                    elif "step" in data.lower():
                        line_tag = "step"
                    self.log_text.insert(tk.END, data, line_tag)
                    if self.auto_scroll_var.get():
                        self.log_text.see(tk.END)

                elif tag == "progress":
                    self.progress_var.set(data)
                    self.status_var.set(f"Generating... {data:.0f}%")

                elif tag == "error":
                    self.log_text.insert(tk.END, data, "error")
                    if self.auto_scroll_var.get():
                        self.log_text.see(tk.END)

                elif tag == "done":
                    self.progress_var.set(100 if data else 0)
                    sep = "=" * 60
                    if data:
                        self.log_text.insert(tk.END,
                            f"\n{sep}\n  Generation complete!\n"
                            f"  Saved: {data}\n{sep}\n", "success")
                        self.status_var.set(f"Complete: {os.path.basename(data)}")
                        self.notebook.select(1)
                        self._show_preview(data)
                    else:
                        self.log_text.insert(tk.END,
                            f"\n{sep}\n  Generation failed.\n{sep}\n", "error")
                        self.status_var.set("Failed")

                    self._set_gen(False)
                    self.process = None

        except Exception:
            pass
        self.root.after(50, self._process_queue)

    def _stop_generation(self):
        if not self.process:
            return
        try:
            if sys.platform == "win32":
                self.process.terminate()
            else:
                self.process.send_signal(signal.SIGINT)
            self.log_text.insert(tk.END, "\n\n[Stopped by user]\n", "warning")
            self.status_var.set("Stopped")
            time.sleep(0.5)
            if self.process and self.process.poll() is None:
                self.process.kill()
            self._set_gen(False)
            self.process = None
        except Exception:
            self._force_shutdown()

    def _force_shutdown(self):
        try:
            if self.process:
                try:
                    self.process.kill()
                except Exception:
                    pass
                if sys.platform == "win32":
                    try:
                        os.system(f"taskkill /F /PID {self.process.pid}")
                    except Exception:
                        pass
            self.log_text.insert(tk.END, "\n\n[FORCED SHUTDOWN]\n", "error")
            self._set_gen(False)
            self.process = None
            self.status_var.set("Force shutdown complete")
            if messagebox.askyesno("Done", "Force shutdown complete. Exit?"):
                self.root.destroy()
                sys.exit(0)
        except Exception as e:
            messagebox.showerror("Critical", f"Shutdown failed: {e}")
            self.root.destroy()
            sys.exit(1)

    # ── Config save/load ──────────────────────────────────────
    def _save_config(self):
        config = {
            "model": self.selected_model.get(),
            "sampler": self.sampler_var.get(),
            "clip_on_cpu": self.clip_cpu_var.get(),
            "flash_attn": self.flash_attn_var.get(),
            "vae_tiling": self.vae_tiling_var.get(),
            "offload_to_cpu": self.offload_cpu_var.get(),
            "parameters": {k: e.get() for k, e in self.params.items()},
            "prompt": self.prompt_text.get(1.0, tk.END).strip(),
            "negative_prompt": self.neg_prompt_text.get(1.0, tk.END).strip(),
        }
        fp = filedialog.asksaveasfilename(
            title="Save config", defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if fp:
            try:
                with open(fp, "w", encoding="utf-8") as fh:
                    json.dump(config, fh, indent=2)
                self.status_var.set(f"Saved: {Path(fp).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {e}")

    def _load_config(self):
        fp = filedialog.askopenfilename(
            title="Load config",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not fp:
            return
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                config = json.load(fh)

            if "model" in config and config["model"] in self.available_models:
                self.selected_model.set(config["model"])
                self._on_model_change()

            if "sampler" in config:
                self.sampler_var.set(config["sampler"])
            if "clip_on_cpu" in config:
                self.clip_cpu_var.set(config["clip_on_cpu"])
            if "flash_attn" in config:
                self.flash_attn_var.set(config["flash_attn"])
            if "vae_tiling" in config:
                self.vae_tiling_var.set(config["vae_tiling"])
            if "offload_to_cpu" in config:
                self.offload_cpu_var.set(config["offload_to_cpu"])

            if "parameters" in config:
                for k, v in config["parameters"].items():
                    if k in self.params:
                        self.params[k].delete(0, tk.END)
                        self.params[k].insert(0, v)

            if "prompt" in config:
                self.prompt_text.delete(1.0, tk.END)
                self.prompt_text.insert(1.0, config["prompt"])
            if "negative_prompt" in config:
                self.neg_prompt_text.delete(1.0, tk.END)
                self.neg_prompt_text.insert(1.0, config["negative_prompt"])

            self.status_var.set(f"Loaded: {Path(fp).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Load failed: {e}")


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) > 1:
        print("Run without arguments to launch GUI.")
        sys.exit(1)
    root = tk.Tk()
    SDGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()