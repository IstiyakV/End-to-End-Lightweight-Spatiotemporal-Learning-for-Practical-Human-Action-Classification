"""Training Configuration — polished form with grouped sections and tabs."""

import customtkinter as ctk
from pathlib import Path
import sys

from gui.settings import load_settings, save_setting
from gui.theme import COLORS

class TrainingConfigFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=COLORS["bg"], corner_radius=0)
        self.app = app

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=30, pady=(25, 0))
        ctk.CTkLabel(header, text="Training Configuration", font=("Segoe UI", 26, "bold"), text_color=COLORS["text"]).pack(anchor="w")
        ctk.CTkLabel(header, text="Configure hyperparameters and start training", font=("Segoe UI", 12), text_color=COLORS["text_dim"]).pack(anchor="w")

        # Tabview
        self.tabview = ctk.CTkTabview(self, fg_color=COLORS["bg"], 
                                      segmented_button_selected_color=COLORS["accent"], 
                                      segmented_button_selected_hover_color=COLORS["accent_hover"],
                                      text_color=COLORS["text"])
        self.tabview.pack(fill="both", expand=True, padx=20, pady=(10, 20))
        
        self.tab_ucf = self.tabview.add("UCF-101 Dataset")
        self.tab_kin = self.tabview.add("Kinetics-700 Dataset")

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from har import config as hcfg

        self.ucf_vars = {}
        self.kin_vars = {}

        settings = load_settings()

        # Load saved paths or fallback to defaults
        ucf_ds = settings.get("ucf101_ds_path", str(hcfg.UCF101_DIR))
        kin_ds = settings.get("kinetics_ds_path", str(hcfg.DATA_DIR))
        ucf_cache = settings.get("ucf101_cache_path", str(hcfg.PROJECT_ROOT / "cache" / "ucf101"))
        kin_cache = settings.get("kinetics_cache_path", str(hcfg.PROJECT_ROOT / "cache" / "kinetics"))

        self._build_form(self.tab_ucf, self.ucf_vars, "ucf101", "ucf101_run", ucf_ds, ucf_cache, True)
        self._build_form(self.tab_kin, self.kin_vars, "kinetics", "kinetics_run", kin_ds, kin_cache, False)

    def _make_section(self, parent, title):
        ctk.CTkFrame(parent, height=1, fg_color=COLORS["border"]).pack(fill="x", padx=20, pady=(10, 5))
        ctk.CTkLabel(parent, text=title, font=("Segoe UI", 12, "bold"),
                     text_color=COLORS["accent"]).pack(anchor="w", padx=20, pady=(5, 8))

    def _param_row(self, parent, label, widget_factory):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=3)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 11), text_color=COLORS["text"],
                     width=110, anchor="w").pack(side="left")
        w = widget_factory(row)
        w.pack(side="right", fill="x", expand=True, padx=(5, 0))
        return w

    def _make_card(self, parent, row, col):
        card = ctk.CTkFrame(parent, fg_color=COLORS["card"], corner_radius=14,
                            border_width=1, border_color=COLORS["card_border"])
        card.grid(row=row, column=col, sticky="nsew", padx=(20 if col == 0 else 10,
                  20 if col == 1 else 10), pady=10)
        return card

    def _slider_with_value(self, parent, from_, to, steps, var, fmt="d"):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        slider = ctk.CTkSlider(frame, from_=from_, to=to, number_of_steps=steps,
                                variable=var, progress_color=COLORS["accent"],
                                button_color=COLORS["accent"], button_hover_color=COLORS["accent_hover"])
        slider.pack(side="left", fill="x", expand=True)
        val_lbl = ctk.CTkLabel(frame, text=f"{var.get():{fmt}}", font=("Segoe UI", 11, "bold"),
                               text_color=COLORS["accent"], width=40)
        val_lbl.pack(side="right", padx=(8, 0))
        var.trace_add("write", lambda *_: val_lbl.configure(text=f"{var.get():{fmt}}"))
        return frame

    def _browse_entry(self, parent, string_var, setting_key):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        entry = ctk.CTkEntry(frame, textvariable=string_var, font=("Segoe UI", 10), 
                             fg_color=COLORS["input_bg"], border_color=COLORS["input_border"])
        entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        def save(*args):
            save_setting(setting_key, string_var.get())
        string_var.trace_add("write", save)
        
        def browse():
            d = ctk.filedialog.askdirectory(initialdir=string_var.get() or ".")
            if d: 
                string_var.set(d)
                # save is triggered automatically by the trace
        ctk.CTkButton(frame, text="...", width=30, fg_color=COLORS["border"], hover_color=COLORS["card_border"], command=browse).pack(side="right")
        return frame

    def _build_form(self, parent_tab, vars_dict, dataset_type, default_exp, default_ds_path, default_cache_path, default_cache_on):
        scroll = ctk.CTkScrollableFrame(parent_tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        scroll.grid_columnconfigure((0, 1), weight=1)

        vars_dict["dataset_type"] = dataset_type

        # -- Left Card (Model) --
        card1 = self._make_card(scroll, 0, 0)
        ctk.CTkLabel(card1, text="\u2630  Experiment & Paths", font=("Segoe UI", 15, "bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=20, pady=(18, 5))

        vars_dict["exp_name"] = ctk.StringVar(value=default_exp)
        self._param_row(card1, "Experiment", lambda p: ctk.CTkEntry(
            p, textvariable=vars_dict["exp_name"], font=("Segoe UI", 11),
            fg_color=COLORS["input_bg"], border_color=COLORS["input_border"], corner_radius=8))

        self._make_section(card1, "LOCATIONS")
        
        vars_dict["dataset_path"] = ctk.StringVar(value=default_ds_path)
        self._param_row(card1, "MP4 Folder", lambda p: self._browse_entry(p, vars_dict["dataset_path"], f"{dataset_type}_ds_path"))

        vars_dict["use_cache"] = ctk.BooleanVar(value=default_cache_on)
        self._param_row(card1, "Use Cache", lambda p: ctk.CTkSwitch(
            p, variable=vars_dict["use_cache"], text="Enable (Lightning fast)",
            progress_color=COLORS["success"], font=("Segoe UI", 10)))
            
        vars_dict["cache_path"] = ctk.StringVar(value=default_cache_path)
        self._param_row(card1, "Cache Folder", lambda p: self._browse_entry(p, vars_dict["cache_path"], f"{dataset_type}_cache_path"))

        self._make_section(card1, "INPUT PARAMETERS")

        vars_dict["img_size"] = ctk.StringVar(value="224")
        self._param_row(card1, "Image Size", lambda p: ctk.CTkSegmentedButton(
            p, values=["112", "128", "160", "224"], variable=vars_dict["img_size"],
            font=("Segoe UI", 11), fg_color=COLORS["input_bg"],
            selected_color=COLORS["accent"], selected_hover_color=COLORS["accent_hover"]))

        vars_dict["n_frames"] = ctk.IntVar(value=10)
        self._param_row(card1, "Frames/Video", lambda p: self._slider_with_value(
            p, 5, 20, 15, vars_dict["n_frames"]))

        vars_dict["frame_step"] = ctk.IntVar(value=15)
        self._param_row(card1, "Frame Step", lambda p: self._slider_with_value(
            p, 5, 30, 25, vars_dict["frame_step"]))

        ctk.CTkFrame(card1, height=15, fg_color="transparent").pack()

        # -- Right Card (Training) --
        card2 = self._make_card(scroll, 0, 1)
        ctk.CTkLabel(card2, text="\u2699  Training Parameters", font=("Segoe UI", 15, "bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=20, pady=(18, 10))

        vars_dict["epochs"] = ctk.StringVar(value="100")
        self._param_row(card2, "Epochs", lambda p: ctk.CTkEntry(
            p, textvariable=vars_dict["epochs"], font=("Segoe UI", 11), width=80,
            fg_color=COLORS["input_bg"], border_color=COLORS["input_border"], corner_radius=8))

        vars_dict["batch"] = ctk.StringVar(value="32")
        self._param_row(card2, "Batch Size", lambda p: ctk.CTkSegmentedButton(
            p, values=["4", "8", "16", "32", "64"], variable=vars_dict["batch"],
            font=("Segoe UI", 11), fg_color=COLORS["input_bg"],
            selected_color=COLORS["accent"], selected_hover_color=COLORS["accent_hover"]))

        vars_dict["lr"] = ctk.StringVar(value="0.0001")
        self._param_row(card2, "Learning Rate", lambda p: ctk.CTkEntry(
            p, textvariable=vars_dict["lr"], font=("Segoe UI", 11), width=100,
            fg_color=COLORS["input_bg"], border_color=COLORS["input_border"], corner_radius=8))
            
        vars_dict["dropout"] = ctk.DoubleVar(value=0.3)
        self._param_row(card2, "Dropout", lambda p: self._slider_with_value(
            p, 0.0, 0.7, 14, vars_dict["dropout"], fmt=".2f"))

        self._make_section(card2, "ARCHITECTURE")

        # Build architecture options: presets first, then user-saved
        try:
            from har.model_builder import (
                preset_architectures, list_saved_architectures,
                build_from_config, count_params, PRESET_META,
            )
            presets = list(preset_architectures().keys())
            saved = list_saved_architectures()
            # Mark saved ones to distinguish from presets
            saved_labels = [f"📄 {s}" for s in saved] if saved else []
            arch_options = presets + saved_labels
            _ARCH_PRESETS = set(presets)
            _ARCH_SAVED_MAP = {f"📄 {s}": s for s in saved}
        except Exception:
            arch_options = ["Paper Default"]
            _ARCH_PRESETS = {"Paper Default"}
            _ARCH_SAVED_MAP = {}

        vars_dict["architecture"] = ctk.StringVar(value="Paper Default")

        # Architecture dropdown
        self._param_row(card2, "Network", lambda p: ctk.CTkOptionMenu(
            p, variable=vars_dict["architecture"], values=arch_options,
            font=("Segoe UI", 11), fg_color=COLORS["input_bg"],
            button_color=COLORS["accent"], corner_radius=8,
            command=lambda _: self._update_arch_desc(vars_dict)))

        # Description label showing params + type
        self._arch_desc_label = ctk.CTkLabel(
            card2, text="", font=("Segoe UI", 9),
            text_color=COLORS["text_dim"], anchor="w")
        self._arch_desc_label.pack(fill="x", padx=24, pady=(0, 6))
        self._arch_presets = _ARCH_PRESETS
        self._arch_saved_map = _ARCH_SAVED_MAP
        # Show initial description
        self.after(100, lambda: self._update_arch_desc(vars_dict))

        self._make_section(card2, "PERFORMANCE")

        vars_dict["amp"] = ctk.BooleanVar(value=True)
        self._param_row(card2, "Mixed Precision", lambda p: ctk.CTkSwitch(
            p, variable=vars_dict["amp"], text="FP16 (halves VRAM)",
            progress_color=COLORS["success"], font=("Segoe UI", 10)))

        vars_dict["workers"] = ctk.StringVar(value="0")
        self._param_row(card2, "Data Workers", lambda p: ctk.CTkSegmentedButton(
            p, values=["0", "2", "4", "8"], variable=vars_dict["workers"],
            font=("Segoe UI", 11), fg_color=COLORS["input_bg"],
            selected_color=COLORS["accent"], selected_hover_color=COLORS["accent_hover"]))

        self._make_section(card2, "RESUME")

        from har import config as hcfg
        checkpoints = ["None"] + [p.name for p in hcfg.CHECKPOINT_DIR.glob("*.pth")] if hcfg.CHECKPOINT_DIR.exists() else ["None"]
        vars_dict["resume"] = ctk.StringVar(value="None")
        self._param_row(card2, "Checkpoint", lambda p: ctk.CTkOptionMenu(
            p, variable=vars_dict["resume"], values=checkpoints,
            font=("Segoe UI", 11), fg_color=COLORS["input_bg"],
            button_color=COLORS["accent"], corner_radius=8))

        ctk.CTkFrame(card2, height=15, fg_color="transparent").pack()

        # -- Action Bar --
        bar = ctk.CTkFrame(scroll, fg_color=COLORS["card"], corner_radius=14,
                           border_width=1, border_color=COLORS["card_border"])
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=(10, 20))

        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.pack(fill="x", padx=25, pady=15)

        # GPU
        from gui.services.gpu_service import detect_gpus
        try:
            gpus = detect_gpus()
            gpu_options = [f"GPU {g['id']}: {g['name']}" for g in gpus] + ["CPU"]
        except:
            gpu_options = ["CPU"]
            
        vars_dict["gpu"] = ctk.StringVar(value=gpu_options[0] if gpu_options else "CPU")

        ctk.CTkLabel(inner, text="\u2622  Device:", font=("Segoe UI", 12, "bold"),
                     text_color=COLORS["text"]).pack(side="left", padx=(0, 8))
        ctk.CTkOptionMenu(inner, variable=vars_dict["gpu"], values=gpu_options,
                          font=("Segoe UI", 11), fg_color=COLORS["input_bg"],
                          button_color=COLORS["accent"], width=280, corner_radius=8
                          ).pack(side="left", padx=(0, 30))

        # Start button
        ctk.CTkButton(inner, text="\u25B6  Start Training", font=("Segoe UI", 15, "bold"),
                      fg_color=COLORS["success"], hover_color=COLORS["success"], height=48,
                      corner_radius=12, width=220, command=lambda: self._start_training(vars_dict)
                      ).pack(side="right")

        # Status
        vars_dict["status_label"] = ctk.CTkLabel(bar, text="", font=("Segoe UI", 11),
                                          text_color=COLORS["text_dim"])
        vars_dict["status_label"].pack(anchor="w", padx=25, pady=(0, 10))

    def _update_arch_desc(self, vars_dict):
        """Update the architecture description label when selection changes."""
        name = vars_dict["architecture"].get()
        try:
            from har.model_builder import (
                preset_architectures, build_from_config, count_params, PRESET_META,
            )
            presets = preset_architectures()
            if name in presets:
                arch = presets[name]
                model = build_from_config(arch, num_classes=101)
                total, _ = count_params(model)
                meta = PRESET_META.get(name, {})
                badge = meta.get("badge", "")
                desc = meta.get("description", "")
                self._arch_desc_label.configure(
                    text=f"{badge}  ·  {total:,} params  ·  {desc}")
            elif name in getattr(self, '_arch_saved_map', {}):
                self._arch_desc_label.configure(text="📄 Saved architecture from Network Architect")
            else:
                self._arch_desc_label.configure(text="")
        except Exception:
            self._arch_desc_label.configure(text="")

    def _get_config(self, vars_dict):
        from gui.services.trainer_service import TrainingConfig
        from har import config as hcfg

        gpu_str = vars_dict["gpu"].get()
        device = f"cuda:{gpu_str.split(':')[0].replace('GPU ', '')}" if "GPU" in gpu_str else "cpu"

        resume = vars_dict["resume"].get()
        resume_path = str(hcfg.CHECKPOINT_DIR / resume) if resume != "None" else None

        # Resolve architecture name — strip emoji prefix from saved architectures
        arch_name = vars_dict["architecture"].get() if "architecture" in vars_dict else "Paper Default"
        saved_map = getattr(self, '_arch_saved_map', {})
        if arch_name in saved_map:
            arch_name = saved_map[arch_name]  # strip "📄 " prefix

        return TrainingConfig(
            dataset_path=vars_dict["dataset_path"].get(), 
            cache_path=vars_dict["cache_path"].get(),
            use_cache=vars_dict["use_cache"].get(),
            experiment_name=vars_dict["exp_name"].get(),
            img_size=int(vars_dict["img_size"].get()),
            n_frames=vars_dict["n_frames"].get(),
            frame_step=vars_dict["frame_step"].get(),
            batch_size=int(vars_dict["batch"].get()),
            epochs=int(vars_dict["epochs"].get()),
            learning_rate=float(vars_dict["lr"].get()),
            dropout=vars_dict["dropout"].get(),
            use_amp=vars_dict["amp"].get(),
            num_workers=int(vars_dict["workers"].get()),
            device=device, resume_checkpoint=resume_path,
            architecture=arch_name,
        )

    def _start_training(self, vars_dict):
        try:
            config = self._get_config(vars_dict)
            self.app.trainer_service.start(config)
            vars_dict["status_label"].configure(text="Training started! Switching to Monitor...",
                                         text_color=COLORS["success"])
            self.after(500, lambda: self.app._switch_frame("monitor"))
        except Exception as e:
            vars_dict["status_label"].configure(text=f"Error: {e}", text_color=COLORS["error"])
