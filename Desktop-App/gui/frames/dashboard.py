"""Dashboard — premium overview with stat cards, status indicators, quick actions."""

import customtkinter as ctk
from pathlib import Path
import sys

from gui.theme import COLORS
def _stat_card(parent, title, value, subtitle="", accent_color="#6c5ce7", row=0, col=0):
    """Reusable stat card widget."""
    card = ctk.CTkFrame(parent, fg_color=COLORS["card"], corner_radius=14,
                         border_width=1, border_color=COLORS["card_border"])
    card.grid(row=row, column=col, sticky="nsew", padx=10, pady=8)

    # Accent top strip
    strip = ctk.CTkFrame(card, height=3, fg_color=accent_color, corner_radius=2)
    strip.pack(fill="x", padx=20, pady=(12, 0))

    ctk.CTkLabel(card, text=title, font=("Segoe UI", 11),
                 text_color=COLORS["text_dim"]).pack(anchor="w", padx=20, pady=(10, 2))

    ctk.CTkLabel(card, text=str(value), font=("Segoe UI", 26, "bold"),
                 text_color=COLORS["text"]).pack(anchor="w", padx=20, pady=(0, 2))

    if subtitle:
        ctk.CTkLabel(card, text=subtitle, font=("Segoe UI", 10),
                     text_color=COLORS["text_dim"]).pack(anchor="w", padx=20, pady=(0, 12))
    else:
        ctk.CTkFrame(card, height=12, fg_color="transparent").pack()

    return card


class DashboardFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=COLORS["bg"], corner_radius=0)
        self.app = app
        self.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # ── Header ──
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, columnspan=4, sticky="ew", padx=30, pady=(25, 5))

        ctk.CTkLabel(header_frame, text="Dashboard",
                     font=("Segoe UI", 26, "bold"), text_color=COLORS["text"]
                     ).pack(side="left")

        ctk.CTkLabel(header_frame, text="Human Action Recognition Control Center",
                     font=("Segoe UI", 12), text_color=COLORS["text_dim"]
                     ).pack(side="left", padx=(15, 0), pady=(8, 0))

        # ── Stat cards row ──
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from har import config as hcfg

        # GPU
        gpu_name = "CPU"
        vram = "N/A"
        try:
            from gui.services.gpu_service import detect_gpus
            gpus = detect_gpus()
            if gpus:
                gpu_name = gpus[0]["name"].replace("NVIDIA GeForce ", "")
                vram = f"{gpus[0]['vram_mb']} MB VRAM"
        except:
            pass

        _stat_card(self, "GPU", gpu_name, vram, "#6c5ce7", row=1, col=0)

        # Datasets
        n_classes = 0
        n_videos = 0
        if hcfg.DATA_DIR.exists():
            dirs = [d for d in hcfg.DATA_DIR.iterdir() if d.is_dir()]
            n_classes = len(dirs)
            for d in dirs:
                n_videos += sum(1 for f in d.iterdir() if f.suffix.lower() in hcfg.VIDEO_EXTENSIONS)

        _stat_card(self, "CLASSES", str(n_classes), f"{n_videos:,} videos", "#00b894", row=1, col=1)

        # Models
        n_models = len(list(hcfg.CHECKPOINT_DIR.glob("*_best.pth"))) if hcfg.CHECKPOINT_DIR.exists() else 0
        _stat_card(self, "MODELS", str(n_models), "trained checkpoints", "#a29bfe", row=1, col=2)

        # Cache
        cache_dir = hcfg.PROJECT_ROOT / "cache" / "kinetics"
        cached = sum(1 for _ in cache_dir.rglob("*.npy")) if cache_dir.exists() else 0
        cache_status = f"{cached:,} cached" if cached else "not cached"
        _stat_card(self, "CACHE", cache_status, "preprocessed tensors",
                   "#00b894" if cached else "#e17055", row=1, col=3)

        # ── Quick Actions ──
        actions_card = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=14,
                                     border_width=1, border_color=COLORS["card_border"])
        actions_card.grid(row=2, column=0, columnspan=4, sticky="ew", padx=40, pady=(15, 10))

        ctk.CTkLabel(actions_card, text="Quick Actions",
                     font=("Segoe UI", 14, "bold"), text_color=COLORS["text"]
                     ).pack(anchor="w", padx=25, pady=(18, 12))

        btn_row = ctk.CTkFrame(actions_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=25, pady=(0, 18))

        actions = [
            ("Start Training", COLORS["accent"], ("#ffffff", "#ffffff"), lambda: app._switch_frame("training")),
            ("Test Model", COLORS["success"], ("#ffffff", "#ffffff"), lambda: app._switch_frame("tester")),
            ("View Results", COLORS["accent_hover"], ("#ffffff", "#ffffff"), lambda: app._switch_frame("results")),
            ("Browse Datasets", COLORS["sidebar_hover"], COLORS["text"], lambda: app._switch_frame("datasets")),
        ]
        for text, color, t_color, cmd in actions:
            ctk.CTkButton(btn_row, text=text, font=("Segoe UI", 12, "bold"),
                          fg_color=color, hover_color=color, text_color=t_color, corner_radius=10,
                          height=42, width=180, command=cmd).pack(side="left", padx=8)

        # ── Training Status ──
        status_card = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=14,
                                    border_width=1, border_color=COLORS["card_border"])
        status_card.grid(row=3, column=0, columnspan=4, sticky="ew", padx=40, pady=(5, 10))

        status_row = ctk.CTkFrame(status_card, fg_color="transparent")
        status_row.pack(fill="x", padx=25, pady=15)

        status = app.trainer_service.progress.status
        status_colors = {"running": COLORS["success"], "paused": COLORS["warning"],
                         "idle": COLORS["text_dim"], "completed": COLORS["accent"],
                         "error": COLORS["error"]}
        dot_color = status_colors.get(status, COLORS["text_dim"])

        # Status dot
        dot = ctk.CTkFrame(status_row, width=10, height=10, fg_color=dot_color, corner_radius=5)
        dot.pack(side="left", padx=(0, 8))

        ctk.CTkLabel(status_row, text=f"Training: {status.upper()}",
                     font=("Segoe UI", 13, "bold"), text_color=dot_color).pack(side="left")

        # ── Trained Models List ──
        models_card = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=14,
                                    border_width=1, border_color=COLORS["card_border"])
        models_card.grid(row=4, column=0, columnspan=4, sticky="nsew", padx=40, pady=(5, 20))

        ctk.CTkLabel(models_card, text="Trained Models",
                     font=("Segoe UI", 14, "bold"), text_color=COLORS["text"]
                     ).pack(anchor="w", padx=25, pady=(15, 8))

        checkpoints = list(hcfg.CHECKPOINT_DIR.glob("*_best.pth")) if hcfg.CHECKPOINT_DIR.exists() else []
        if checkpoints:
            import torch
            for cp in checkpoints[:6]:
                try:
                    ckpt = torch.load(str(cp), map_location="cpu", weights_only=False)
                    acc = ckpt.get("val_accuracy", 0)
                    epoch = ckpt.get("epoch", "?")
                    n_cls = ckpt.get("config", {}).get("num_classes", "?")
                    row = ctk.CTkFrame(models_card, fg_color=COLORS["input_bg"], corner_radius=8, height=36)
                    row.pack(fill="x", padx=20, pady=2)
                    row.pack_propagate(False)
                    ctk.CTkLabel(row, text=f"  {cp.stem}", font=("Segoe UI", 11),
                                 text_color=COLORS["text"]).pack(side="left", padx=8)
                    ctk.CTkLabel(row, text=f"acc: {acc:.2%}  |  epoch: {epoch}  |  {n_cls} classes",
                                 font=("Segoe UI", 10), text_color=COLORS["text_dim"]
                                 ).pack(side="right", padx=12)
                except:
                    pass
        else:
            ctk.CTkLabel(models_card, text="  No trained models yet. Go to Training to start.",
                         font=("Segoe UI", 11), text_color=COLORS["text_dim"]
                         ).pack(anchor="w", padx=25, pady=(0, 15))

        ctk.CTkFrame(models_card, height=10, fg_color="transparent").pack()
