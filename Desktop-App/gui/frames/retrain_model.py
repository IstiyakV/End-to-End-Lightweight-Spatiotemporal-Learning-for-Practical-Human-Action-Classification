"""
Retrain Model Frame — polished twin-card interface to load checkpoints,
extract nested metadata, adjust hyperparameters, and resume/fine-tune training.
"""

import customtkinter as ctk
from pathlib import Path
import sys
import torch

from gui.settings import load_settings, save_setting
from gui.theme import COLORS


class RetrainModelFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=COLORS["bg"], corner_radius=0)
        self.app = app

        # Add project root to sys.path to access har modules
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from har import config as hcfg
        self.hcfg = hcfg

        # Variables
        self.vars = {
            "checkpoint_path": ctk.StringVar(value=""),
            "exp_name": ctk.StringVar(value="retrain_experiment"),
            "strategy": ctk.StringVar(value="Freeze Conv Layers (Classifier Only)"),
            "target_epochs": ctk.StringVar(value="30"),
            "learning_rate": ctk.StringVar(value="0.00001"),
            "dataset_path": ctk.StringVar(value=""),
            "use_cache": ctk.BooleanVar(value=True),
            "cache_path": ctk.StringVar(value=""),
            "workers": ctk.StringVar(value="0" if sys.platform == "win32" else "4"),
            "device": ctk.StringVar(value="CPU"),
            "amp": ctk.BooleanVar(value=True),
        }

        # Extracted metadata storage
        self.extracted_metadata = {
            "epoch": "—",
            "val_acc": "—",
            "img_size": "—",
            "n_frames": "—",
            "frame_step": "—",
            "num_classes": "—",
            "classes": [],
            "architecture": "—",
        }

        # Set default dataset and cache paths from settings
        settings = load_settings()
        self.vars["dataset_path"].set(settings.get("ucf101_ds_path", str(hcfg.UCF101_DIR)))
        self.vars["cache_path"].set(settings.get("ucf101_cache_path", str(hcfg.PROJECT_ROOT / "cache" / "ucf101")))

        # Main Layout
        self._build_ui()

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=30, pady=(25, 5))
        ctk.CTkLabel(header, text="Retrain & Fine-Tune Model", font=("Segoe UI", 26, "bold"), text_color=COLORS["text"]).pack(anchor="w")
        ctk.CTkLabel(header, text="Load PyTorch checkpoints, extract parameters, and fine-tune classifier heads or full networks",
                     font=("Segoe UI", 12), text_color=COLORS["text_dim"]).pack(anchor="w")

        # Scrollable container for the forms
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=20, pady=(5, 10))
        scroll.grid_columnconfigure((0, 1), weight=1)

        # -- Left Card: Checkpoint Selector & Metadata Grid --
        self.card_left = self._make_card(scroll, 0, 0)
        ctk.CTkLabel(self.card_left, text="📂  Model Checkpoint Info", font=("Segoe UI", 16, "bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=20, pady=(18, 5))

        # Checkpoint Selection Row
        row_select = ctk.CTkFrame(self.card_left, fg_color="transparent")
        row_select.pack(fill="x", padx=20, pady=8)
        
        ctk.CTkLabel(row_select, text="Checkpoint file:", font=("Segoe UI", 11), text_color=COLORS["text"], width=100, anchor="w").pack(side="left")
        
        # Populate available checkpoint files dynamically
        self.ckpt_dropdown = ctk.CTkOptionMenu(
            row_select, variable=self.vars["checkpoint_path"], values=[],
            font=("Segoe UI", 11), fg_color=COLORS["input_bg"],
            button_color=COLORS["accent"], corner_radius=8, width=220,
            command=self._on_checkpoint_selected
        )
        self.ckpt_dropdown.pack(side="left", fill="x", expand=True, padx=(5, 5))
        
        ctk.CTkButton(row_select, text="Browse...", width=70, fg_color=COLORS["border"], hover_color=COLORS["card_border"], command=self._browse_checkpoint).pack(side="right")

        # Metadata extraction trigger button
        self.btn_extract = ctk.CTkButton(
            self.card_left, text="🔍 Extract Checkpoint Details", font=("Segoe UI", 12, "bold"),
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], height=35, corner_radius=8,
            command=self._extract_metadata
        )
        self.btn_extract.pack(fill="x", padx=20, pady=(5, 15))

        self._make_section(self.card_left, "EXTRACTED PARAMETERS")

        # Stats Grid for extracted checkpoint parameters
        self.grid_frame = ctk.CTkFrame(self.card_left, fg_color=COLORS["input_bg"], border_width=1, border_color=COLORS["border"], corner_radius=10)
        self.grid_frame.pack(fill="x", padx=20, pady=5)
        self.grid_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.stat_labels = {}
        stats_to_create = [
            ("Original Epoch", "epoch", 0, 0),
            ("Val Accuracy", "val_acc", 0, 2),
            ("Image Size", "img_size", 1, 0),
            ("Frame Count", "n_frames", 1, 2),
            ("Frame Step", "frame_step", 2, 0),
            ("Classes Count", "num_classes", 2, 2),
        ]

        for label_text, key, row, col in stats_to_create:
            # Title cell
            cell_title = ctk.CTkFrame(self.grid_frame, fg_color="transparent")
            cell_title.grid(row=row*2, column=col, sticky="w", padx=15, pady=(8, 0))
            ctk.CTkLabel(cell_title, text=label_text, font=("Segoe UI", 9, "bold"), text_color=COLORS["text_dim"]).pack()

            # Value cell
            cell_val = ctk.CTkFrame(self.grid_frame, fg_color="transparent")
            cell_val.grid(row=row*2+1, column=col, columnspan=2 if col==0 or col==2 else 1, sticky="w", padx=15, pady=(0, 8))
            lbl = ctk.CTkLabel(cell_val, text="—", font=("Segoe UI", 14, "bold"), text_color=COLORS["accent"])
            lbl.pack()
            self.stat_labels[key] = lbl

        # Extra metadata row for architecture
        arch_row = ctk.CTkFrame(self.card_left, fg_color="transparent")
        arch_row.pack(fill="x", padx=20, pady=(15, 5))
        ctk.CTkLabel(arch_row, text="Detected Architecture: ", font=("Segoe UI", 11, "bold"), text_color=COLORS["text_dim"]).pack(side="left")
        self.lbl_detected_arch = ctk.CTkLabel(arch_row, text="—", font=("Segoe UI", 11, "bold"), text_color=COLORS["text"])
        self.lbl_detected_arch.pack(side="left", padx=5)

        # Classes box
        self._make_section(self.card_left, "CLASSES DETECTED")
        self.classes_textbox = ctk.CTkTextbox(self.card_left, font=("Consolas", 10), height=85,
                                              fg_color=COLORS["input_bg"], border_width=1, border_color=COLORS["border"], corner_radius=8)
        self.classes_textbox.pack(fill="x", padx=20, pady=(0, 15))
        self.classes_textbox.insert("1.0", "No checkpoint loaded yet.\nExtract metadata to see configured target classes.")
        self.classes_textbox.configure(state="disabled")

        # -- Right Card: Retraining Hyperparameters --
        self.card_right = self._make_card(scroll, 0, 1)
        ctk.CTkLabel(self.card_right, text="⚙️  Retraining Hyperparameters", font=("Segoe UI", 16, "bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=20, pady=(18, 10))

        # Training Strategy Choice
        self._param_row(self.card_right, "Training Strategy", lambda p: ctk.CTkSegmentedButton(
            p, values=["Fine-Tune Classifier", "Full Retraining"], variable=self.vars["strategy"],
            font=("Segoe UI", 11), fg_color=COLORS["input_bg"],
            selected_color=COLORS["accent"], selected_hover_color=COLORS["accent_hover"],
            command=self._on_strategy_changed
        ))

        # Explanatory tip label
        self.lbl_strategy_tip = ctk.CTkLabel(
            self.card_right, text="❄️ Recommended. Freezes conv blocks. Highly stable & fast fine-tuning.",
            font=("Segoe UI", 9), text_color=COLORS["success"], anchor="w"
        )
        self.lbl_strategy_tip.pack(fill="x", padx=24, pady=(0, 8))

        # Target Epochs Offset
        self._param_row(self.card_right, "Target Epochs", lambda p: ctk.CTkEntry(
            p, textvariable=self.vars["target_epochs"], font=("Segoe UI", 11), width=80,
            fg_color=COLORS["input_bg"], border_color=COLORS["input_border"], corner_radius=8
        ))
        ctk.CTkLabel(self.card_right, text="Specify total target epochs (e.g. if loaded checkpoint is epoch 10, setting 40 adds 30 epochs)",
                     font=("Segoe UI", 9), text_color=COLORS["text_dim"], anchor="w").pack(fill="x", padx=24, pady=(0, 8))

        # Custom Learning Rate
        self._param_row(self.card_right, "Learning Rate", lambda p: ctk.CTkEntry(
            p, textvariable=self.vars["learning_rate"], font=("Segoe UI", 11), width=100,
            fg_color=COLORS["input_bg"], border_color=COLORS["input_border"], corner_radius=8
        ))
        self.lbl_lr_tip = ctk.CTkLabel(self.card_right, text="Typical: 1e-5 for fine-tuning, 1e-4 for full model training",
                                       font=("Segoe UI", 9), text_color=COLORS["text_dim"], anchor="w")
        self.lbl_lr_tip.pack(fill="x", padx=24, pady=(0, 8))

        self._make_section(self.card_right, "RETRAINING LOCATIONS")
        
        self._param_row(self.card_right, "Dataset MP4s", lambda p: self._browse_entry(p, self.vars["dataset_path"], "ucf101_ds_path"))

        self._param_row(self.card_right, "Use Cache", lambda p: ctk.CTkSwitch(
            p, variable=self.vars["use_cache"], text="Enable fast .npy loading",
            progress_color=COLORS["success"], font=("Segoe UI", 10)
        ))
            
        self._param_row(self.card_right, "Cache Folder", lambda p: self._browse_entry(p, self.vars["cache_path"], "ucf101_cache_path"))

        self._make_section(self.card_right, "PERFORMANCE")

        self._param_row(self.card_right, "Mixed Precision", lambda p: ctk.CTkSwitch(
            p, variable=self.vars["amp"], text="AMP FP16 (conserves VRAM)",
            progress_color=COLORS["success"], font=("Segoe UI", 10)
        ))

        self._param_row(self.card_right, "Data Workers", lambda p: ctk.CTkSegmentedButton(
            p, values=["0", "2", "4", "8"], variable=self.vars["workers"],
            font=("Segoe UI", 11), fg_color=COLORS["input_bg"],
            selected_color=COLORS["accent"], selected_hover_color=COLORS["accent_hover"]
        ))

        # Set initial segmented selections
        self.vars["strategy"].set("Fine-Tune Classifier")

        # -- Action Bar (Bottom) --
        bar = ctk.CTkFrame(scroll, fg_color=COLORS["card"], corner_radius=14,
                           border_width=1, border_color=COLORS["card_border"])
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=(10, 20))

        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.pack(fill="x", padx=25, pady=15)

        # GPU / Device Detection
        from gui.services.gpu_service import detect_gpus
        try:
            gpus = detect_gpus()
            gpu_options = [f"GPU {g['id']}: {g['name']}" for g in gpus] + ["CPU"]
        except:
            gpu_options = ["CPU"]
        self.vars["device"].set(gpu_options[0] if gpu_options else "CPU")

        ctk.CTkLabel(inner, text="⚡ Target Device:", font=("Segoe UI", 12, "bold"),
                     text_color=COLORS["text"]).pack(side="left", padx=(0, 8))
        ctk.CTkOptionMenu(inner, variable=self.vars["device"], values=gpu_options,
                          font=("Segoe UI", 11), fg_color=COLORS["input_bg"],
                          button_color=COLORS["accent"], width=260, corner_radius=8
                          ).pack(side="left", padx=(0, 30))

        # Start button
        self.btn_start = ctk.CTkButton(
            inner, text="🔄 Start Retraining", font=("Segoe UI", 15, "bold"),
            fg_color=COLORS["success"], hover_color=COLORS["success"], height=48,
            corner_radius=12, width=220, command=self._start_retraining
        )
        self.btn_start.pack(side="right")

        # Status label
        self.lbl_status = ctk.CTkLabel(bar, text="", font=("Segoe UI", 11), text_color=COLORS["text_dim"])
        self.lbl_status.pack(anchor="w", padx=25, pady=(0, 10))

        # Dynamic search for checkpoints on startup
        self.after(200, self._scan_checkpoints)

    def _make_card(self, parent, row, col):
        card = ctk.CTkFrame(parent, fg_color=COLORS["card"], corner_radius=14,
                            border_width=1, border_color=COLORS["card_border"])
        card.grid(row=row, column=col, sticky="nsew", padx=(20 if col == 0 else 10,
                  20 if col == 1 else 10), pady=10)
        return card

    def _make_section(self, parent, title):
        ctk.CTkFrame(parent, height=1, fg_color=COLORS["border"]).pack(fill="x", padx=20, pady=(12, 6))
        ctk.CTkLabel(parent, text=title, font=("Segoe UI", 11, "bold"),
                     text_color=COLORS["accent"]).pack(anchor="w", padx=20, pady=(4, 6))

    def _param_row(self, parent, label, widget_factory):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=4)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 11), text_color=COLORS["text"],
                     width=115, anchor="w").pack(side="left")
        w = widget_factory(row)
        w.pack(side="right", fill="x", expand=True, padx=(5, 0))
        return w

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
        ctk.CTkButton(frame, text="...", width=30, fg_color=COLORS["border"], hover_color=COLORS["card_border"], command=browse).pack(side="right")
        return frame

    def _scan_checkpoints(self):
        """Scans CHECKPOINT_DIR for saved weights."""
        if not self.hcfg.CHECKPOINT_DIR.exists():
            self.hcfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        files = sorted(list(self.hcfg.CHECKPOINT_DIR.glob("*.pth")), key=lambda p: p.stat().st_mtime, reverse=True)
        paths = [str(f.resolve()) for f in files]
        display_names = [f.name for f in files]

        if paths:
            self.ckpt_dropdown.configure(values=display_names)
            self.vars["checkpoint_path"].set(display_names[0])
            self.ckpt_paths_map = dict(zip(display_names, paths))
            self.lbl_status.configure(text=f"Scan complete. Found {len(paths)} local checkpoints.", text_color=COLORS["text_dim"])
        else:
            self.ckpt_dropdown.configure(values=["No Checkpoints Found"])
            self.vars["checkpoint_path"].set("No Checkpoints Found")
            self.ckpt_paths_map = {}
            self.lbl_status.configure(text="No checkpoints (.pth) found in results/checkpoints. Click Browse... to load custom weights.", text_color="#d63031")

    def _browse_checkpoint(self):
        f = ctk.filedialog.askopenfilename(filetypes=[("PyTorch Checkpoints", "*.pth")])
        if f:
            p = Path(f)
            self.ckpt_paths_map[p.name] = str(p.resolve())
            current_vals = list(self.ckpt_dropdown.cget("values"))
            if p.name not in current_vals:
                current_vals.append(p.name)
                self.ckpt_dropdown.configure(values=current_vals)
            self.vars["checkpoint_path"].set(p.name)
            self._on_checkpoint_selected(p.name)

    def _on_checkpoint_selected(self, choice):
        self.lbl_status.configure(text=f"Selected checkpoint: {choice}. Click Extract Details to read parameters.", text_color=COLORS["text_dim"])

    def _on_strategy_changed(self, choice):
        if choice == "Fine-Tune Classifier":
            self.lbl_strategy_tip.configure(text="❄️ Recommended. Freezes conv blocks. Highly stable & fast fine-tuning.", text_color=COLORS["success"])
            self.vars["learning_rate"].set("0.00001")
        else:
            self.lbl_strategy_tip.configure(text="🔥 Warning. Keeps all weights trainable. Risky on small datasets/VRAM.", text_color="#fdcb6e")
            self.vars["learning_rate"].set("0.0001")

    def _extract_metadata(self):
        choice = self.vars["checkpoint_path"].get()
        if not choice or choice == "No Checkpoints Found":
            self.lbl_status.configure(text="Error: Select or browse for a valid PyTorch checkpoint first.", text_color="#d63031")
            return

        ckpt_path = self.ckpt_paths_map.get(choice)
        if not ckpt_path or not Path(ckpt_path).exists():
            self.lbl_status.configure(text="Error: Selected checkpoint file path is invalid or missing.", text_color="#d63031")
            return

        self.lbl_status.configure(text="Extracting checkpoint headers...", text_color=COLORS["accent"])
        self.update_idletasks()

        try:
            # Read dictionary containing metadata (safe python load)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            
            # Epochs
            epoch = ckpt.get("epoch", "—")
            self.stat_labels["epoch"].configure(text=str(epoch))
            self.extracted_metadata["epoch"] = epoch
            if isinstance(epoch, int):
                self.vars["target_epochs"].set(str(epoch + 30))
                # Generate a default experiment name appended with retrain
                stem = Path(ckpt_path).stem
                self.vars["exp_name"].set(f"{stem}_retrain")

            # Val accuracy
            acc = ckpt.get("best_val_acc", ckpt.get("val_accuracy", None))
            if acc is not None:
                acc_text = f"{acc:.2%}" if isinstance(acc, float) else str(acc)
                self.stat_labels["val_acc"].configure(text=acc_text)
                self.extracted_metadata["val_acc"] = acc_text
            else:
                self.stat_labels["val_acc"].configure(text="N/A")
                self.extracted_metadata["val_acc"] = "N/A"

            # Parse inner configuration dictionary
            cfg_dict = ckpt.get("config", {})
            
            # Architecture detection
            detected_arch = "Compact3DCNN (Paper)"
            if "conv_filters" in cfg_dict:
                detected_arch = f"Compact3DCNN (Filters: {cfg_dict['conv_filters']})"
            self.lbl_detected_arch.configure(text=detected_arch)
            self.extracted_metadata["architecture"] = detected_arch

            # Image Size
            img = cfg_dict.get("img_size", "—")
            if isinstance(img, (list, tuple)):
                img_text = f"{img[0]}x{img[1]}"
            else:
                img_text = str(img)
            self.stat_labels["img_size"].configure(text=img_text)
            self.extracted_metadata["img_size"] = img_text

            # Frames
            n_frames = cfg_dict.get("n_frames", "—")
            self.stat_labels["n_frames"].configure(text=str(n_frames))
            self.extracted_metadata["n_frames"] = n_frames

            # Frame Step
            step = cfg_dict.get("frame_step", "—")
            self.stat_labels["frame_step"].configure(text=str(step))
            self.extracted_metadata["frame_step"] = step

            # Classes
            classes = ckpt.get("class_names", [])
            self.extracted_metadata["classes"] = classes
            num_cls = cfg_dict.get("num_classes", len(classes) if classes else "—")
            self.stat_labels["num_classes"].configure(text=str(num_cls))
            self.extracted_metadata["num_classes"] = num_cls

            self.classes_textbox.configure(state="normal")
            self.classes_textbox.delete("1.0", "end")
            if classes:
                self.classes_textbox.insert("1.0", f"Classes ({len(classes)}):\n" + ", ".join(classes))
            else:
                self.classes_textbox.insert("1.0", "No class names array saved in checkpoint. Defaults mapping UCF-101 index labels.")
            self.classes_textbox.configure(state="disabled")

            self.lbl_status.configure(text="✓ Checkpoint metadata successfully extracted and synchronized!", text_color=COLORS["success"])

        except Exception as e:
            self.lbl_status.configure(text=f"Extraction failed: {str(e)}", text_color="#d63031")
            import traceback
            traceback.print_exc()

    def _start_retraining(self):
        """Builds configuration and triggers the TrainerService background run."""
        if self.app.trainer_service.is_running:
            self.lbl_status.configure(text="Error: Training service is already running an active task.", text_color="#d63031")
            return

        choice = self.vars["checkpoint_path"].get()
        if not choice or choice == "No Checkpoints Found":
            self.lbl_status.configure(text="Error: Please specify a valid checkpoint file.", text_color="#d63031")
            return

        ckpt_path = self.ckpt_paths_map.get(choice)
        if not ckpt_path or not Path(ckpt_path).exists():
            self.lbl_status.configure(text="Error: Chosen checkpoint path is unreachable.", text_color="#d63031")
            return

        # Hyperparameter Validations
        try:
            epochs = int(self.vars["target_epochs"].get())
            if epochs <= 0:
                raise ValueError
        except:
            self.lbl_status.configure(text="Error: Target epochs must be a positive integer.", text_color="#d63031")
            return

        try:
            lr = float(self.vars["learning_rate"].get())
            if lr <= 0:
                raise ValueError
        except:
            self.lbl_status.configure(text="Error: Learning rate must be a positive float.", text_color="#d63031")
            return

        ds_path = self.vars["dataset_path"].get()
        if not ds_path or not Path(ds_path).exists():
            self.lbl_status.configure(text="Error: Dataset MP4 folder path is missing or invalid.", text_color="#d63031")
            return

        # Build training config object
        from gui.services.trainer_service import TrainingConfig

        cfg = TrainingConfig()
        cfg.experiment_name = self.vars["exp_name"].get().strip() or "retrain_run"
        cfg.resume_checkpoint = ckpt_path
        cfg.epochs = epochs
        cfg.learning_rate = lr
        cfg.dataset_path = ds_path
        cfg.use_cache = self.vars["use_cache"].get()
        cfg.cache_path = self.vars["cache_path"].get()
        cfg.use_amp = self.vars["amp"].get()
        cfg.num_workers = int(self.vars["workers"].get())
        cfg.freeze_conv = (self.vars["strategy"].get() == "Fine-Tune Classifier")

        # Map active hardware device
        dev_choice = self.vars["device"].get()
        if "GPU" in dev_choice:
            # Extract ID e.g. 'GPU 0' -> 'cuda:0'
            try:
                gpu_id = dev_choice.split(":")[0].split(" ")[1]
                cfg.device = f"cuda:{gpu_id}"
            except:
                cfg.device = "cuda:0"
        else:
            cfg.device = "cpu"

        # Transfer frame metadata overrides if extracted
        if self.extracted_metadata["n_frames"] != "—":
            cfg.n_frames = int(self.extracted_metadata["n_frames"])
        if self.extracted_metadata["frame_step"] != "—":
            cfg.frame_step = int(self.extracted_metadata["frame_step"])
        if self.extracted_metadata["img_size"] != "—":
            try:
                cfg.img_size = int(self.extracted_metadata["img_size"].split("x")[0])
            except:
                cfg.img_size = 224

        # Start Training in background thread
        try:
            self.lbl_status.configure(text="Configuring background run...", text_color=COLORS["accent"])
            self.app.trainer_service.start(cfg)
            
            # Switch views immediately to Monitor
            self.after(300, lambda: self.app._switch_frame("monitor"))
        except Exception as e:
            self.lbl_status.configure(text=f"Failed to start: {str(e)}", text_color="#d63031")
