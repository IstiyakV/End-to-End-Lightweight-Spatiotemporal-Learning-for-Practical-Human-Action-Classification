"""
Transfer Learning Frame — standalone dashboard with live matplotlib plots and validation consoles.
"""

import queue
import sys
import tkinter as tk
import customtkinter as ctk
from pathlib import Path
from typing import Optional
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gui.theme import COLORS
from gui.settings import load_settings, save_setting
from gui.services.transfer_learning_service import TransferConfig, TransferStatus, inspect_and_validate_backbone


class TransferLearningFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=COLORS["bg"], corner_radius=0)
        self.app = app
        self._log_queue = queue.Queue()
        self._auto_scroll = True
        self._poll_id = None
        self._last_drawn_epoch = 0

        # Dedicated isolated Service
        from gui.services.transfer_learning_service import TransferLearningService
        self.service = TransferLearningService()

        # UI variables
        self.vars = {
            "weight_path": ctk.StringVar(value=load_settings().get("backbone_weights", "")),
            "backbone_arch": ctk.StringVar(value="r3d_18"),
            "custom_head": ctk.StringVar(value="Paper Default"),
            "dataset_path": ctk.StringVar(value=load_settings().get("ucf101_ds_path", "")),
            "cache_path": ctk.StringVar(value=load_settings().get("ucf101_cache_path", "")),
            "use_cache": ctk.BooleanVar(value=True),
            "batch_size": ctk.StringVar(value="8"),
            "epochs": ctk.StringVar(value="20"),
            "learning_rate": ctk.StringVar(value="0.0001"),
            "optimizer": ctk.StringVar(value="Adam"),
            "scheduler": ctk.StringVar(value="None"),
            "amp": ctk.BooleanVar(value=True),
            "workers": ctk.StringVar(value="0"),
            "model_name": ctk.StringVar(value="transfer_model_run"),
            "device": ctk.StringVar(value="GPU 0: cuda:0"),
        }

        # 2-column layout: Left Column (Control & Setup) | Right Column (Live Analytics)
        self.grid_columnconfigure(0, weight=45, minsize=480)
        self.grid_columnconfigure(1, weight=55, minsize=580)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        
        # Scrollable container for parameters (Left Column)
        self.scroll_left = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_left.grid(row=1, column=0, sticky="nsew", padx=(25, 10), pady=(5, 5))
        self._build_left_column()

        # Right Column (Tabview containing Charts and Log Consoles)
        self._build_right_column()

        # Bottom Telemetry & Progress
        self._build_status_bar()

        # Register callbacks and start redirection
        self.service.add_callback(self._on_progress)
        self._poll_console()

    # ─────────────────────────── HEADER ────────────────────────────
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=25, pady=(15, 5))

        ctk.CTkLabel(
            header, text="⇄  Transfer Learning Dashboard",
            font=("Segoe UI", 22, "bold"), text_color=COLORS["text"]
        ).pack(side="left")

        # Isolated model identity field in header
        lbl_frame = ctk.CTkFrame(header, fg_color="transparent")
        lbl_frame.pack(side="right")
        
        ctk.CTkLabel(lbl_frame, text="🏷️ Model ID:", font=("Segoe UI", 11, "bold"), text_color=COLORS["text_dim"]).pack(side="left", padx=5)
        ctk.CTkEntry(
            lbl_frame, textvariable=self.vars["model_name"], font=("Consolas", 11, "bold"),
            fg_color=COLORS["input_bg"], border_color=COLORS["input_border"], width=180, corner_radius=6
        ).pack(side="left")

    # ──────────────────────── LEFT COLUMN ──────────────────────────
    def _build_left_column(self):
        # 1. pre-trained Ingestion Card
        card_ingest = ctk.CTkFrame(self.scroll_left, fg_color=COLORS["card"], corner_radius=12,
                                   border_width=1, border_color=COLORS["card_border"])
        card_ingest.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(card_ingest, text="📦 1. Pre-trained Backbone Ingestion", 
                     font=("Segoe UI", 13, "bold"), text_color=COLORS["accent"]).pack(anchor="w", padx=15, pady=(12, 6))

        self._param_row(card_ingest, "Weights Path (.pth)", lambda p: self._browse_file(p, self.vars["weight_path"], "backbone_weights"))
        self._param_row(card_ingest, "Backbone Arch", lambda p: ctk.CTkOptionMenu(
            p, variable=self.vars["backbone_arch"], values=["r3d_18", "mc3_18", "r2plus1d_18"],
            font=("Segoe UI", 11), fg_color=COLORS["input_bg"], button_color=COLORS["accent"], corner_radius=6
        ))

        btn_verify = ctk.CTkButton(
            card_ingest, text="🔍 Ingest & Validate Backbone", font=("Segoe UI", 11, "bold"),
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], height=32, corner_radius=8,
            command=self._verify_backbone
        )
        btn_verify.pack(fill="x", padx=15, pady=(6, 12))

        # 2. Hybrid Model Binding Card
        card_bind = ctk.CTkFrame(self.scroll_left, fg_color=COLORS["card"], corner_radius=12,
                                 border_width=1, border_color=COLORS["card_border"])
        card_bind.pack(fill="x", pady=10)

        ctk.CTkLabel(card_bind, text="⚙️ 2. Custom Classification Head", 
                     font=("Segoe UI", 13, "bold"), text_color=COLORS["accent"]).pack(anchor="w", padx=15, pady=(12, 6))

        # Dynamically scan custom heads designed in Architect
        custom_archs = ["Paper Default"]
        from har.model_builder import list_saved_architectures
        try:
            custom_archs += list_saved_architectures()
        except:
            pass

        self._param_row(card_bind, "Custom Head", lambda p: ctk.CTkOptionMenu(
            p, variable=self.vars["custom_head"], values=custom_archs,
            font=("Segoe UI", 11), fg_color=COLORS["input_bg"], button_color=COLORS["accent"], corner_radius=6
        ))
        
        lbl_binding_tip = ctk.CTkLabel(
            card_bind, text="❄️ Auto-Freezing Backbone. Connects your custom head dynamically.",
            font=("Segoe UI", 9), text_color=COLORS["success"], anchor="w"
        )
        lbl_binding_tip.pack(fill="x", padx=18, pady=(0, 10))

        # 3. Caching & Location Card
        card_data = ctk.CTkFrame(self.scroll_left, fg_color=COLORS["card"], corner_radius=12,
                                 border_width=1, border_color=COLORS["card_border"])
        card_data.pack(fill="x", pady=10)

        ctk.CTkLabel(card_data, text="📁 3. Datasets & Space Optimizations", 
                     font=("Segoe UI", 13, "bold"), text_color=COLORS["accent"]).pack(anchor="w", padx=15, pady=(12, 6))

        self._param_row(card_data, "Use Cache", lambda p: ctk.CTkSwitch(
            p, variable=self.vars["use_cache"], text="Load fast .npy vectors",
            progress_color=COLORS["success"], font=("Segoe UI", 10)
        ))
        self._param_row(card_data, "Cache Folder", lambda p: self._browse_dir(p, self.vars["cache_path"], "ucf101_cache_path"))
        self._param_row(card_data, "Raw Videos Path", lambda p: self._browse_dir(p, self.vars["dataset_path"], "ucf101_ds_path"))

        # 4. Hyperparameters Panel
        card_hyp = ctk.CTkFrame(self.scroll_left, fg_color=COLORS["card"], corner_radius=12,
                                border_width=1, border_color=COLORS["card_border"])
        card_hyp.pack(fill="x", pady=10)

        ctk.CTkLabel(card_hyp, text="⚡ 4. Hyperparameter Settings", 
                     font=("Segoe UI", 13, "bold"), text_color=COLORS["accent"]).pack(anchor="w", padx=15, pady=(12, 6))

        self._param_row(card_hyp, "Optimizer", lambda p: ctk.CTkSegmentedButton(
            p, values=["Adam", "SGD"], variable=self.vars["optimizer"],
            font=("Segoe UI", 11), fg_color=COLORS["input_bg"], selected_color=COLORS["accent"], selected_hover_color=COLORS["accent_hover"]
        ))
        self._param_row(card_hyp, "LR Scheduler", lambda p: ctk.CTkOptionMenu(
            p, variable=self.vars["scheduler"], values=["None", "StepLR"],
            font=("Segoe UI", 11), fg_color=COLORS["input_bg"], button_color=COLORS["accent"], corner_radius=6
        ))
        self._param_row(card_hyp, "Learning Rate", lambda p: ctk.CTkEntry(
            p, textvariable=self.vars["learning_rate"], font=("Segoe UI", 11), width=100,
            fg_color=COLORS["input_bg"], border_color=COLORS["input_border"], corner_radius=8
        ))
        self._param_row(card_hyp, "Target Epochs", lambda p: ctk.CTkEntry(
            p, textvariable=self.vars["epochs"], font=("Segoe UI", 11), width=80,
            fg_color=COLORS["input_bg"], border_color=COLORS["input_border"], corner_radius=8
        ))
        self._param_row(card_hyp, "Batch Size", lambda p: ctk.CTkEntry(
            p, textvariable=self.vars["batch_size"], font=("Segoe UI", 11), width=80,
            fg_color=COLORS["input_bg"], border_color=COLORS["input_border"], corner_radius=8
        ))
        self._param_row(card_hyp, "Mixed Precision", lambda p: ctk.CTkSwitch(
            p, variable=self.vars["amp"], text="AMP (saves VRAM)",
            progress_color=COLORS["success"], font=("Segoe UI", 10)
        ))

    # ──────────────────────── RIGHT COLUMN ──────────────────────────
    def _build_right_column(self):
        self.right_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.right_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 25), pady=(5, 5))
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(0, weight=1)

        # Tabview for telemetry & logs
        self.tabs = ctk.CTkTabview(
            self.right_frame, fg_color=COLORS["card"], corner_radius=12,
            border_width=1, border_color=COLORS["card_border"],
            segmented_button_selected_color=COLORS["accent"],
            segmented_button_selected_hover_color=COLORS["accent_hover"],
            segmented_button_fg_color=COLORS["input_bg"]
        )
        self.tabs.grid(row=0, column=0, sticky="nsew")
        
        self.tab_graph = self.tabs.add("📈 Live Analytics Graph")
        self.tab_console = self.tabs.add("🖥️ Output & Ingestion Logs")

        # 1. Matplotlib Analytics Setup
        self._build_charts(self.tab_graph)

        # 2. Console Redirection Setup
        self._build_console_box(self.tab_console)

    def _build_charts(self, parent):
        idx = 1 if ctk.get_appearance_mode().lower() == "dark" else 0
        plot_bg   = COLORS["plot_bg"][idx]
        plot_grid = COLORS["plot_grid"][idx]
        text_dim  = COLORS["text_dim"][idx]
        text_col  = COLORS["text"][idx]

        self.fig = Figure(facecolor=plot_bg, constrained_layout=True)
        self.ax_loss  = self.fig.add_subplot(221)
        self.ax_acc   = self.fig.add_subplot(222)
        self.ax_lr    = self.fig.add_subplot(223)
        self.ax_speed = self.fig.add_subplot(224)

        for ax in (self.ax_loss, self.ax_acc, self.ax_lr, self.ax_speed):
            ax.set_facecolor(plot_bg)
            ax.tick_params(colors=text_dim, labelsize=8)
            ax.spines["bottom"].set_color(plot_grid)
            ax.spines["left"].set_color(plot_grid)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color=plot_grid, linewidth=0.5, alpha=0.6)

        self.ax_loss.set_yscale("log")
        self._apply_chart_titles(text_col)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    def _apply_chart_titles(self, text_col):
        for ax, title in (
            (self.ax_loss,  "Loss (Log Scale)"),
            (self.ax_acc,   "Accuracy (%)"),
            (self.ax_lr,    "Learning Rate"),
            (self.ax_speed, "Epoch Duration (s)"),
        ):
            ax.set_title(title, color=text_col, fontsize=9, fontweight="bold", pad=4)

    def _build_console_box(self, parent):
        idx = 1 if ctk.get_appearance_mode().lower() == "dark" else 0
        bg = COLORS["console_bg"][idx]

        hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hdr.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(hdr, text="● Ingestion Diagnostics & PyTorch Console", font=("Segoe UI", 11, "bold"), text_color=COLORS["text"]).pack(side="left")
        ctk.CTkButton(hdr, text="Clear", width=52, height=24, font=("Segoe UI", 10), corner_radius=6,
                      fg_color=COLORS["border"], text_color=COLORS["text_dim"], hover_color=COLORS["input_bg"],
                      command=self._clear_console).pack(side="right")

        self._console = tk.Text(
            parent, bg=bg, fg="#a0a8b8", font=("Consolas", 10), wrap="word", relief="flat", bd=0,
            insertbackground="#6c5ce7", selectbackground="#252836"
        )
        self._console.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(0, 10))

        sb = ctk.CTkScrollbar(parent, command=self._console.yview)
        sb.pack(side="right", fill="y", padx=(0, 6), pady=(0, 10))
        self._console.configure(yscrollcommand=sb.set)
        self._console.configure(state="disabled")

        # Tag configs
        self._console.tag_config("error",   foreground="#ff6b6b")
        self._console.tag_config("warning", foreground="#fdcb6e")
        self._console.tag_config("epoch",   foreground="#00b894")
        self._console.tag_config("info",    foreground="#74b9ff")
        self._console.tag_config("default", foreground="#a0a8b8")

        # Start capturing standard console stdout redirects
        from gui.services.console_redirector import ConsoleRedirector
        self.redirector = ConsoleRedirector(self._log_queue)
        self.redirector.start()

    # ─────────────────────── STATUS BAR ────────────────────────────
    def _build_status_bar(self):
        bar = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=12,
                           border_width=1, border_color=COLORS["card_border"])
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=25, pady=(5, 15))
        bar.grid_columnconfigure((0,1,2,3,4), weight=1)

        metrics = ctk.CTkFrame(bar, fg_color="transparent")
        metrics.grid(row=0, column=0, columnspan=5, sticky="ew", padx=15, pady=(8, 4))
        metrics.grid_columnconfigure((0,1,2,3,4), weight=1)

        self.status_pill = ctk.CTkLabel(
            metrics, text="  ● IDLE  ", font=("Segoe UI", 11, "bold"),
            fg_color=COLORS["border"], text_color=COLORS["text_dim"], corner_radius=8
        )
        self.status_pill.grid(row=0, column=0, sticky="w", padx=(0, 10))

        for i, (attr, text) in enumerate([
            ("epoch_lbl", "Epoch: -/-"),
            ("batch_lbl", "Batch: -/-"),
            ("best_lbl",  "Best Acc: -"),
            ("eta_lbl",   "ETA: -"),
        ], start=1):
            lbl = ctk.CTkLabel(metrics, text=text, font=("Segoe UI", 11), text_color=COLORS["text"])
            lbl.grid(row=0, column=i, padx=5)
            setattr(self, attr, lbl)

        # Action Buttons frame (Right aligned on status bar)
        btn_frame = ctk.CTkFrame(metrics, fg_color="transparent")
        btn_frame.grid(row=0, column=5, sticky="e")

        self.btn_start = ctk.CTkButton(btn_frame, text="🚀 Train", width=75, height=28, fg_color=COLORS["success"], text_color="#000", font=("Segoe UI", 11, "bold"), corner_radius=6, command=self._start)
        self.btn_start.pack(side="left", padx=2)

        self.btn_pause = ctk.CTkButton(btn_frame, text="⏸ Pause", width=75, height=28, fg_color=COLORS["warning"], text_color="#000", font=("Segoe UI", 11, "bold"), corner_radius=6, command=self._pause)
        self.btn_pause.pack(side="left", padx=2)

        self.btn_resume = ctk.CTkButton(btn_frame, text="▶ Resume", width=80, height=28, fg_color=COLORS["accent"], font=("Segoe UI", 11, "bold"), corner_radius=6, command=self._resume)
        self.btn_resume.pack(side="left", padx=2)

        self.btn_retrain = ctk.CTkButton(btn_frame, text="🔄 Retrain", width=85, height=28, fg_color=COLORS["error"], font=("Segoe UI", 11, "bold"), corner_radius=6, command=self._retrain)
        self.btn_retrain.pack(side="left", padx=2)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(bar, progress_color=COLORS["accent"], fg_color=COLORS["progress_bg"], height=4, corner_radius=2)
        self.progress_bar.grid(row=1, column=0, columnspan=5, sticky="ew", padx=15, pady=(0, 6))
        self.progress_bar.set(0)

    # ────────────────────── PARAMETER ROWS ─────────────────────────
    def _param_row(self, parent, label, widget_factory):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=15, pady=3)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 11), text_color=COLORS["text"], width=130, anchor="w").pack(side="left")
        w = widget_factory(row)
        w.pack(side="right", fill="x", expand=True, padx=(5, 0))
        return w

    def _browse_file(self, parent, string_var, setting_key):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        entry = ctk.CTkEntry(frame, textvariable=string_var, font=("Segoe UI", 10), fg_color=COLORS["input_bg"], border_color=COLORS["input_border"])
        entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        def save(*args):
            save_setting(setting_key, string_var.get())
        string_var.trace_add("write", save)

        def browse():
            f = ctk.filedialog.askopenfilename(filetypes=[("Weights checkpoint", "*.pth")])
            if f:
                string_var.set(f)
        ctk.CTkButton(frame, text="...", width=28, fg_color=COLORS["border"], hover_color=COLORS["card_border"], command=browse).pack(side="right")
        return frame

    def _browse_dir(self, parent, string_var, setting_key):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        entry = ctk.CTkEntry(frame, textvariable=string_var, font=("Segoe UI", 10), fg_color=COLORS["input_bg"], border_color=COLORS["input_border"])
        entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        def save(*args):
            save_setting(setting_key, string_var.get())
        string_var.trace_add("write", save)

        def browse():
            d = ctk.filedialog.askdirectory(initialdir=string_var.get() or ".")
            if d:
                string_var.set(d)
        ctk.CTkButton(frame, text="...", width=28, fg_color=COLORS["border"], hover_color=COLORS["card_border"], command=browse).pack(side="right")
        return frame

    # ────────────────── INGESTION & DIAGNOSTICS ──────────────────────
    def _verify_backbone(self):
        """Dynamic pre-trained ingestion verification loop."""
        path = self.vars["weight_path"].get().strip()
        arch = self.vars["backbone_arch"].get()

        self._console.configure(state="normal")
        self._console.insert("end", "\n" + "="*50 + "\n")
        self._console.configure(state="disabled")

        if not path or not Path(path).exists():
            self._append_to_console("❌ Error: Select a valid pre-trained weights file path first.\n", "error")
            self.tabs.set("🖥️ Output & Ingestion Logs")
            return

        self._append_to_console(f"⏳ Verification triggered for {arch} checkpoint...\n", "info")
        self.tabs.set("🖥️ Output & Ingestion Logs")
        
        # Load safely on CPU
        inspection = inspect_and_validate_backbone(path, arch)
        self._append_to_console(inspection["log"] + "\n", "default")
        
        if "Suitable" in inspection["status"]:
            self._append_to_console(f"💡 Features successfully matched! feature_dim = {inspection['feature_dim']}\n", "epoch")
        else:
            self._append_to_console(f"⚠️ Validation warnings: {inspection['message']}\n", "warning")

    # ─────────────────────── CONTROLS ──────────────────────────────
    def _start(self):
        if self.service.is_running:
            self._append_to_console("⚠️ Trainer is already running an active task.\n", "warning")
            return

        cfg = self._build_config()
        if not cfg: return

        self._append_to_console("🚀 Starting background Transfer Learning training thread...\n", "info")
        self.service.start(cfg)

    def _pause(self):
        self.service.pause()

    def _resume(self):
        self.service.resume()

    def _retrain(self):
        cfg = self._build_config()
        if not cfg: return

        self._append_to_console("🔄 Retrain triggered. Resetting all historical metrics...\n", "error")
        self.service.retrain(cfg)

    def _build_config(self) -> Optional[TransferConfig]:
        """Validate input hyperparameters and construct TransferConfig."""
        try:
            epochs = int(self.vars["epochs"].get())
            batch = int(self.vars["batch_size"].get())
            lr = float(self.vars["learning_rate"].get())
            if epochs <= 0 or batch <= 0 or lr <= 0: raise ValueError
        except:
            self._append_to_console("❌ Error: Epochs/Batch/LR must be positive numerical values.\n", "error")
            return None

        cfg = TransferConfig()
        cfg.model_name = self.vars["model_name"].get().strip() or "transfer_run"
        cfg.backbone_weights_path = self.vars["weight_path"].get()
        cfg.backbone_architecture = self.vars["backbone_arch"].get()
        cfg.custom_head_name = self.vars["custom_head"].get()
        cfg.dataset_path = self.vars["dataset_path"].get()
        cfg.cache_path = self.vars["cache_path"].get()
        cfg.use_cache = self.vars["use_cache"].get()
        cfg.batch_size = batch
        cfg.epochs = epochs
        cfg.learning_rate = lr
        cfg.optimizer_type = self.vars["optimizer"].get()
        cfg.scheduler_type = self.vars["scheduler"].get()
        cfg.use_amp = self.vars["amp"].get()
        cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        return cfg

    # ────────────────────── TELEMETRY CALLBACKS ─────────────────────
    def _on_progress(self, progress):
        self.after(0, self._update_ui, progress)

    def _update_ui(self, progress):
        idx = 1 if ctk.get_appearance_mode().lower() == "dark" else 0
        s = progress.status

        STATUS_FG = {
            "running":   ("#00b894", "#00b894"),
            "paused":    ("#fdcb6e", "#fdcb6e"),
            "stopped":   ("#e17055", "#e17055"),
            "completed": ("#6c5ce7", "#6c5ce7"),
            "error":     ("#e17055", "#e17055"),
            "idle":      COLORS["text_dim"],
        }
        STATUS_BG = {
            "running":   ("#d0f5ea", "#0d3b2e"),
            "paused":    ("#fef9e7", "#3d2e00"),
            "stopped":   ("#fdecea", "#3d0d0d"),
            "completed": ("#ede9fe", "#1e1040"),
            "error":     ("#fdecea", "#3d0d0d"),
            "idle":      COLORS["border"],
        }
        fg = STATUS_FG.get(s, COLORS["text_dim"])
        bg = STATUS_BG.get(s, COLORS["border"])
        if isinstance(fg, tuple): fg = fg[idx]
        if isinstance(bg, tuple): bg = bg[idx]

        self.status_pill.configure(text=f"  ● {s.upper()}  ", text_color=fg, fg_color=bg)
        self.epoch_lbl.configure(text=f"Epoch: {progress.current_epoch}/{progress.total_epochs}")
        
        if progress.total_batches > 0 and s not in ["paused", "stopped", "completed", "idle"]:
            self.batch_lbl.configure(text=f"Batch: {progress.current_batch}/{progress.total_batches}")
        else:
            self.batch_lbl.configure(text="Batch: -/-")

        self.best_lbl.configure(text=f"Best Acc: {progress.best_val_acc:.2%}")
        
        mins = int(progress.eta_seconds // 60)
        self.eta_lbl.configure(text=f"ETA: {mins // 60}h {mins % 60}m")

        if progress.total_epochs > 0:
            self.progress_bar.set(progress.current_epoch / progress.total_epochs)

        # Plot Redraws on Epoch changes
        h = progress.history
        num_points = len(h.get("train_loss", []))
        if num_points == 0:
            if self._last_drawn_epoch > 0:
                self._clear_charts()
                self._last_drawn_epoch = 0
        else:
            if self._last_drawn_epoch != num_points:
                self._redraw_charts(progress)
                self._last_drawn_epoch = num_points

    def _clear_charts(self):
        idx = 1 if ctk.get_appearance_mode().lower() == "dark" else 0
        plot_bg   = COLORS["plot_bg"][idx]
        plot_grid = COLORS["plot_grid"][idx]
        text_dim  = COLORS["text_dim"][idx]
        text_col  = COLORS["text"][idx]

        for ax in (self.ax_loss, self.ax_acc, self.ax_lr, self.ax_speed):
            ax.clear()
            ax.set_facecolor(plot_bg)
            ax.tick_params(colors=text_dim, labelsize=8)
            ax.spines["bottom"].set_color(plot_grid)
            ax.spines["left"].set_color(plot_grid)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color=plot_grid, linewidth=0.5, alpha=0.6)
            
        self.ax_loss.set_yscale("log")
        self._apply_chart_titles(text_col)
        self.canvas.draw_idle()

    def _redraw_charts(self, progress):
        h   = progress.history
        idx = 1 if ctk.get_appearance_mode().lower() == "dark" else 0
        plot_bg   = COLORS["plot_bg"][idx]
        plot_grid = COLORS["plot_grid"][idx]
        text_dim  = COLORS["text_dim"][idx]
        text_col  = COLORS["text"][idx]

        C_TRAIN = "#6c5ce7"
        C_VAL   = "#00b894"
        C_LR    = "#f39c12"
        C_SPEED = "#9b59b6"

        n = len(h["train_loss"])
        epochs = list(range(1, n + 1))

        for ax in (self.ax_loss, self.ax_acc, self.ax_lr, self.ax_speed):
            ax.clear()
            ax.set_facecolor(plot_bg)
            ax.tick_params(colors=text_dim, labelsize=8)
            ax.spines["bottom"].set_color(plot_grid)
            ax.spines["left"].set_color(plot_grid)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color=plot_grid, linewidth=0.5, alpha=0.6)

        # 1. Loss
        self.ax_loss.plot(epochs, h["train_loss"], color=C_TRAIN, linewidth=1.8, label="Train")
        self.ax_loss.plot(epochs, h["val_loss"], color=C_VAL, linewidth=1.8, label="Val")
        self.ax_loss.fill_between(epochs, h["train_loss"], alpha=0.08, color=C_TRAIN)
        self.ax_loss.fill_between(epochs, h["val_loss"], alpha=0.08, color=C_VAL)
        self.ax_loss.set_yscale("log")
        self.ax_loss.legend(facecolor=plot_bg, edgecolor=plot_grid, labelcolor=text_col, fontsize=8)

        # 2. Accuracy
        ta = [v * 100 for v in h["train_acc"]]
        va = [v * 100 for v in h["val_acc"]]
        self.ax_acc.plot(epochs, ta, color=C_TRAIN, linewidth=1.8, label="Train")
        self.ax_acc.plot(epochs, va, color=C_VAL, linewidth=1.8, label="Val")
        self.ax_acc.fill_between(epochs, ta, alpha=0.08, color=C_TRAIN)
        self.ax_acc.fill_between(epochs, va, alpha=0.08, color=C_VAL)
        self.ax_acc.set_ylabel("%", color=text_dim, fontsize=8)
        self.ax_acc.legend(facecolor=plot_bg, edgecolor=plot_grid, labelcolor=text_col, fontsize=8)

        # 3. Learning Rate
        if h.get("lr"):
            lrs = h["lr"][:n]
            self.ax_lr.plot(epochs[:len(lrs)], lrs, color=C_LR, linewidth=1.8)
            self.ax_lr.fill_between(epochs[:len(lrs)], lrs, alpha=0.10, color=C_LR)

        # 4. Speed
        if h.get("epoch_time"):
            times = h["epoch_time"][:n]
            self.ax_speed.plot(epochs[:len(times)], times, color=C_SPEED, linewidth=1.8)
            self.ax_speed.fill_between(epochs[:len(times)], times, alpha=0.10, color=C_SPEED)

        self._apply_chart_titles(text_col)
        self.canvas.draw_idle()

    # ─────────────────── CONSOLE REDIRECTS ─────────────────────────
    def _poll_console(self):
        chunks = []
        try:
            while True:
                chunks.append(self._log_queue.get_nowait())
        except queue.Empty:
            pass

        if chunks:
            self._append_to_console("".join(chunks))

        self._poll_id = self.after(120, self._poll_console)

    def _append_to_console(self, text: str, tag: str = ""):
        self._console.configure(state="normal")
        for line in text.splitlines(keepends=True):
            line_tag = tag
            if not line_tag:
                lo = line.lower()
                if any(k in lo for k in ("error", "exception", "failed")):
                    line_tag = "error"
                elif any(k in lo for k in ("warning", "warn")):
                    line_tag = "warning"
                elif "verdict:" in lo or "epoch" in lo:
                    line_tag = "epoch"
                elif any(k in lo for k in ("verification", "loading", "inspecting")):
                    line_tag = "info"
                else:
                    line_tag = "default"
            self._console.insert("end", line, line_tag)
        self._console.configure(state="disabled")
        if self._auto_scroll:
            self._console.see("end")

    def _clear_console(self):
        self._console.configure(state="normal")
        self._console.delete("1.0", "end")
        self._console.configure(state="disabled")

    def on_theme_changed(self, mode):
        self.configure(fg_color=COLORS["bg"])
        idx = 1 if mode.lower() == "dark" else 0
        
        # 1. Update standard console bg
        self._console.configure(bg=COLORS["console_bg"][idx])
        
        # 2. Update syntax tag colors
        TAG_COLORS = {
            "error":   ("#dc2626", "#ff6b6b"),
            "warning": ("#d97706", "#fdcb6e"),
            "epoch":   ("#16a34a", "#00b894"),
            "best":    ("#b45309", "#ffd700"),
            "info":    ("#2563eb", "#74b9ff"),
            "default": ("#475569", "#a0a8b8"),
        }
        for tag, colors in TAG_COLORS.items():
            self._console.tag_config(tag, foreground=colors[idx])
            
        # 3. Dynamic plot redraw
        h = self.service.progress.history
        if h.get("train_loss"):
            self._redraw_charts(self.service.progress)
        else:
            self._clear_charts()

    # ──────────────────────── CLEANUP ──────────────────────────────
    def destroy(self):
        if self._poll_id:
            self.after_cancel(self._poll_id)
        if hasattr(self, "redirector"):
            self.redirector.stop()
        self.service.stop()
        super().destroy()
