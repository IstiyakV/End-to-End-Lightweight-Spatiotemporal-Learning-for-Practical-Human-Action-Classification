"""Results Viewer — training curves, confusion matrix, metrics, export, and checkpoint management."""

import customtkinter as ctk
import json
import time
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import threading

from gui.theme import COLORS


class ResultsViewerFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=COLORS["bg"], corner_radius=0)
        self.app = app
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from har import config as hcfg
        self.hcfg = hcfg

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(25, 10))

        ctk.CTkLabel(header, text="Results & Storage", font=("Segoe UI", 24, "bold"),
                     text_color=COLORS["text"]).pack(side="left")

        # Tabview across entire row 1
        self.tabview = ctk.CTkTabview(self, fg_color=COLORS["card"], 
                                      segmented_button_selected_color=COLORS["accent"], 
                                      segmented_button_selected_hover_color="#7c6cf7",
                                      text_color=COLORS["text"])
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=30, pady=(0, 15))
        
        self.tab_charts = self.tabview.add("Training Curves & Metrics")
        self.tab_checkpoints = self.tabview.add("Unfinished Checkpoint Manager")

        # ────────────────────────────────────────────────────────────────
        # TAB 1: CHARTS & METRICS
        # ────────────────────────────────────────────────────────────────
        self.tab_charts.grid_columnconfigure(0, weight=1)
        self.tab_charts.grid_rowconfigure(1, weight=1)

        charts_header = ctk.CTkFrame(self.tab_charts, fg_color="transparent")
        charts_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(5, 5))

        ctk.CTkLabel(charts_header, text="Select Experiment:", font=("Segoe UI", 12, "bold"),
                     text_color=COLORS["accent"]).pack(side="left", padx=(5, 10))

        experiments = self._find_experiments()
        self.exp_var = ctk.StringVar(value=experiments[0] if experiments else "None")
        self.exp_menu = ctk.CTkOptionMenu(charts_header, variable=self.exp_var, values=experiments or ["None"],
                                          command=self._load_results, font=("Segoe UI", 11),
                                          fg_color=COLORS["input_bg"], button_color=COLORS["accent"])
        self.exp_menu.pack(side="left")

        ctk.CTkButton(charts_header, text="Export Figures", width=120, height=32,
                      fg_color=COLORS["accent"], font=("Segoe UI", 11),
                      corner_radius=8, command=self._export).pack(side="right", padx=5)

        chart_card = ctk.CTkFrame(self.tab_charts, fg_color=COLORS["bg"], corner_radius=10,
                                  border_width=1, border_color=COLORS["border"])
        chart_card.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 5))

        idx = 1 if ctk.get_appearance_mode().lower() == "dark" else 0
        plot_bg = COLORS["plot_bg"][idx]

        self.fig = Figure(figsize=(14, 8), facecolor=plot_bg)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        # Info
        self.info_lbl = ctk.CTkLabel(self.tab_charts, text="Select an experiment to view results",
                                      font=("Segoe UI", 11), text_color=COLORS["text_dim"])
        self.info_lbl.grid(row=2, column=0, padx=10, pady=(5, 10))

        # Banner for on-demand metrics generation (initially hidden)
        self.metrics_gen_banner = ctk.CTkFrame(self.tab_charts, fg_color=COLORS["card"], corner_radius=10,
                                               border_width=1, border_color=COLORS["border"])
        
        self.banner_inner = ctk.CTkFrame(self.metrics_gen_banner, fg_color="transparent")
        self.banner_inner.pack(fill="x", padx=15, pady=10)
        
        self.banner_lbl = ctk.CTkLabel(self.banner_inner, text="📊 This experiment is missing per-class F1 metrics. Click to generate now.",
                                       font=("Segoe UI", 11, "bold"), text_color=COLORS["text"])
        self.banner_lbl.pack(side="left", padx=(0, 15))
        
        self.btn_gen_metrics = ctk.CTkButton(self.banner_inner, text="⚡ Generate Per-Class F1 Metrics", width=220, height=32,
                                             fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                                             font=("Segoe UI", 11, "bold"), text_color=("#000", "#000"), corner_radius=8,
                                             command=self._start_metrics_generation)
        self.btn_gen_metrics.pack(side="right")
        
        # Progress Bar & Label (nested inside banner, hidden initially)
        self.progress_frame = ctk.CTkFrame(self.metrics_gen_banner, fg_color="transparent")
        
        self.gen_progress_bar = ctk.CTkProgressBar(self.progress_frame, progress_color=COLORS["accent"], fg_color=COLORS["progress_bg"], height=6, corner_radius=3)
        self.gen_progress_bar.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.gen_progress_bar.set(0)
        
        self.gen_progress_lbl = ctk.CTkLabel(self.progress_frame, text="Preparing evaluation...", font=("Segoe UI", 10), text_color=COLORS["text_dim"])
        self.gen_progress_lbl.pack(side="right")

        # ────────────────────────────────────────────────────────────────
        # TAB 2: CHECKPOINT MANAGER
        # ────────────────────────────────────────────────────────────────
        self.tab_checkpoints.grid_columnconfigure(0, weight=1)
        self.tab_checkpoints.grid_rowconfigure(1, weight=1)

        # Manager Header Row
        m_header = ctk.CTkFrame(self.tab_checkpoints, fg_color="transparent")
        m_header.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 5))

        ctk.CTkLabel(m_header, text="Clean Temporary Checkpoints", font=("Segoe UI", 14, "bold"),
                     text_color=COLORS["accent"]).pack(side="left")

        self.btn_clear_all = ctk.CTkButton(m_header, text="🗑 Clear All Unfinished Checkpoints",
                                           fg_color=COLORS["error"], hover_color="#c0392b",
                                           font=("Segoe UI", 11, "bold"), text_color="#ffffff",
                                           corner_radius=8, command=self._clear_all_unfinished)
        self.btn_clear_all.pack(side="right")

        # Scrollable container list
        self.list_scroll = ctk.CTkScrollableFrame(self.tab_checkpoints, fg_color=COLORS["bg"],
                                                 corner_radius=10, border_width=1, border_color=COLORS["border"],
                                                 scrollbar_button_color=COLORS["border"],
                                                 scrollbar_button_hover_color=COLORS["card_border"])
        self.list_scroll.grid(row=1, column=0, sticky="nsew", padx=15, pady=(5, 10))
        self.list_scroll.grid_columnconfigure(0, weight=1)

        # Initial load triggers
        if experiments:
            self._load_results(experiments[0])
            
        # Hook tab show to refresh checkpoints automatically when tab is opened
        self.tabview.configure(command=self._on_tab_switched)
        self._refresh_unfinished_checkpoints()

    def _on_tab_switched(self):
        """Trigger update on active tab view."""
        active = self.tabview.get()
        if active == "Unfinished Checkpoint Manager":
            self._refresh_unfinished_checkpoints()
        elif active == "Training Curves & Metrics":
            # Refresh dropdown in case new metrics finished
            exps = self._find_experiments()
            current = self.exp_var.get()
            self.exp_menu.configure(values=exps or ["None"])
            if exps and current not in exps:
                self.exp_var.set(exps[0])
                self._load_results(exps[0])

    def _find_experiments(self):
        exps = set()
        if self.hcfg.METRICS_DIR.exists():
            for f in self.hcfg.METRICS_DIR.glob("*_metrics.json"):
                exps.add(f.stem.replace("_metrics", ""))
        if self.hcfg.CHECKPOINT_DIR.exists():
            for f in self.hcfg.CHECKPOINT_DIR.glob("*_training_state.json"):
                exps.add(f.stem.replace("_training_state", ""))
            for f in self.hcfg.CHECKPOINT_DIR.glob("*_state.json"):
                if not f.name.endswith("_training_state.json"):
                    exps.add(f.stem.replace("_state", ""))
        return sorted(exps) if exps else []

    def _load_results(self, exp_name):
        if exp_name == "None":
            return
        self.fig.clear()
        try:
            state_path = self.hcfg.CHECKPOINT_DIR / f"{exp_name}_training_state.json"
            if not state_path.exists():
                state_path = self.hcfg.CHECKPOINT_DIR / f"{exp_name}_state.json"
            metrics_path = self.hcfg.METRICS_DIR / f"{exp_name}_metrics.json"

            has_history = state_path.exists()
            has_metrics = metrics_path.exists()

            # Dynamic show/hide of per-class F1 metric generator banner
            if has_history and not has_metrics:
                self.metrics_gen_banner.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))
                self.banner_inner.pack(fill="x", padx=15, pady=10)
                self.progress_frame.pack_forget()
                self.btn_gen_metrics.configure(state="normal")
            else:
                self.metrics_gen_banner.grid_forget()

            if not has_history and not has_metrics:
                self.info_lbl.configure(text=f"No results for '{exp_name}'")
                self.canvas.draw()
                return

            n_plots = (2 if has_history else 0) + (1 if has_metrics else 0)
            plot_idx = 1

            idx = 1 if ctk.get_appearance_mode().lower() == "dark" else 0
            plot_bg = COLORS["plot_bg"][idx]
            plot_grid = COLORS["plot_grid"][idx]
            text_dim = COLORS["text_dim"][idx]
            text_color = COLORS["text"][idx]
            c_accent = COLORS["accent"][idx]
            c_success = COLORS["success"][idx]
            
            if has_history:
                with open(state_path) as f:
                    state = json.load(f)
                if "history" in state:
                    history = state["history"]
                else:
                    history = state

                epochs = range(1, len(history.get("train_loss", [])) + 1)

                # Loss plot
                ax1 = self.fig.add_subplot(1, n_plots, plot_idx)
                ax1.plot(epochs, history.get("train_loss", []), color=c_accent, label="Train", linewidth=1.5)
                ax1.plot(epochs, history.get("val_loss", []), color=c_success, label="Val", linewidth=1.5)
                ax1.set_title("Loss", color=text_color)
                ax1.legend(facecolor=plot_bg, edgecolor=plot_grid, labelcolor=text_color)
                self._style_ax(ax1)
                plot_idx += 1

                # Accuracy plot
                ax2 = self.fig.add_subplot(1, n_plots, plot_idx)
                train_acc = history.get("train_accuracy", history.get("train_acc", []))
                val_acc = history.get("val_accuracy", history.get("val_acc", []))
                ax2.plot(epochs, train_acc, color=c_accent, label="Train", linewidth=1.5)
                ax2.plot(epochs, val_acc, color=c_success, label="Val", linewidth=1.5)
                ax2.set_title("Accuracy", color=text_color)
                ax2.legend(facecolor=plot_bg, edgecolor=plot_grid, labelcolor=text_color)
                self._style_ax(ax2)
                plot_idx += 1

            if has_metrics:
                with open(metrics_path) as f:
                    metrics = json.load(f)

                # Per-class F1 bar chart (top N)
                per_class = metrics.get("per_class", {})
                if per_class:
                    sorted_cls = sorted(per_class.items(), key=lambda x: x[1]["f1"], reverse=True)[:20]
                    names = [c[0] for c in sorted_cls]
                    f1s = [c[1]["f1"] for c in sorted_cls]

                    ax3 = self.fig.add_subplot(1, n_plots, plot_idx)
                    bars = ax3.barh(names, f1s, color=c_accent, alpha=0.85)
                    ax3.set_title(f"Top-20 F1 (acc={metrics['accuracy']:.2%})", color=text_color)
                    ax3.set_xlim(0, 1.05)
                    ax3.invert_yaxis()
                    self._style_ax(ax3)

                info = f"Accuracy: {metrics['accuracy']:.2%} | Macro F1: {metrics['macro']['f1']:.4f}"
                self.info_lbl.configure(text=info, text_color=COLORS["success"])
            elif has_history:
                best_val_acc = state.get("best_val_acc", state.get("best_acc", 0.0))
                if isinstance(best_val_acc, float):
                    acc_text = f"{best_val_acc:.2%}"
                else:
                    acc_text = str(best_val_acc)
                info = f"Status: {state.get('status', 'unknown').upper()}  |  Best Training/Val Accuracy: {acc_text}  |  Epochs Run: {state.get('current_epoch', len(history.get('train_loss', [])))}"
                self.info_lbl.configure(text=info, text_color=COLORS["success"])

            self.fig.tight_layout(pad=2)
        except Exception as e:
            self.info_lbl.configure(text=f"Failed to plot results: {e}", text_color=COLORS["warning"])
        finally:
            self.canvas.draw()

    def _style_ax(self, ax):
        idx = 1 if ctk.get_appearance_mode().lower() == "dark" else 0
        plot_bg = COLORS["plot_bg"][idx]
        plot_grid = COLORS["plot_grid"][idx]
        text_dim = COLORS["text_dim"][idx]

        ax.set_facecolor(plot_bg)
        ax.tick_params(colors=text_dim, labelsize=8)
        ax.spines['bottom'].set_color(plot_grid)
        ax.spines['left'].set_color(plot_grid)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _export(self):
        exp = self.exp_var.get()
        if exp == "None":
            return
        out = self.hcfg.FIGURES_DIR / f"{exp}_gui_export.png"
        idx = 1 if ctk.get_appearance_mode().lower() == "dark" else 0
        plot_bg = COLORS["plot_bg"][idx]
        self.fig.savefig(str(out), dpi=300, facecolor=plot_bg)
        self.info_lbl.configure(text=f"Exported: {out}", text_color=COLORS["success"])

    # ────────────────────────────────────────────────────────────────
    # UNFINISHED CHECKPOINT SCANNING & MANAGEMENT
    # ────────────────────────────────────────────────────────────────
    def _find_unfinished_checkpoints(self) -> list:
        """Scan CHECKPOINT_DIR for _latest.pth, _pause.pth, and their training_state files."""
        checkpoints_map = {}
        
        if not self.hcfg.CHECKPOINT_DIR.exists():
            return []

        # 1. Scan for paused or interrupted json states
        for f in self.hcfg.CHECKPOINT_DIR.glob("*_training_state.json"):
            exp_name = f.name.replace("_training_state.json", "")
            try:
                with open(f) as fh:
                    state = json.load(fh)
                
                status = state.get("status", "unknown")
                epoch = state.get("current_epoch", 0)
                best_acc = state.get("best_val_acc", state.get("best_acc", "—"))
                modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime))
                
                checkpoints_map[exp_name] = {
                    "name": exp_name,
                    "status": status,
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "modified": modified,
                    "files": [f],
                    "size_bytes": f.stat().st_size
                }
            except:
                pass

        # 2. Scan for latest.pth files
        for f in self.hcfg.CHECKPOINT_DIR.glob("*_latest.pth"):
            exp_name = f.name.replace("_latest.pth", "")
            modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime))
            
            if exp_name not in checkpoints_map:
                checkpoints_map[exp_name] = {
                    "name": exp_name,
                    "status": "interrupted",
                    "epoch": "unknown",
                    "best_acc": "—",
                    "modified": modified,
                    "files": [],
                    "size_bytes": 0
                }
            checkpoints_map[exp_name]["files"].append(f)
            checkpoints_map[exp_name]["size_bytes"] += f.stat().st_size

        # 3. Scan for pause.pth files
        for f in self.hcfg.CHECKPOINT_DIR.glob("*_pause.pth"):
            exp_name = f.name.replace("_pause.pth", "")
            modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime))
            
            if exp_name not in checkpoints_map:
                checkpoints_map[exp_name] = {
                    "name": exp_name,
                    "status": "paused",
                    "epoch": "unknown",
                    "best_acc": "—",
                    "modified": modified,
                    "files": [],
                    "size_bytes": 0
                }
            checkpoints_map[exp_name]["files"].append(f)
            checkpoints_map[exp_name]["size_bytes"] += f.stat().st_size

        # Format list and calculate individual component counts
        output_list = []
        for exp_name, data in checkpoints_map.items():
            # If an experiment is marked completed but still has latest/pause weights, list them!
            # Otherwise we focus heavily on interrupted or paused ones
            output_list.append(data)

        # Sort by modification time (newest first)
        return sorted(output_list, key=lambda x: x["modified"], reverse=True)

    def _refresh_unfinished_checkpoints(self):
        """Update and redraw the checkpoint manager list view."""
        for w in self.list_scroll.winfo_children():
            w.destroy()

        data_list = self._find_unfinished_checkpoints()
        
        if not data_list:
            self.btn_clear_all.configure(state="disabled")
            
            lbl_empty = ctk.CTkLabel(self.list_scroll, 
                                     text="🎉 Clean Slate! No unfinished or temporary checkpoint files found.",
                                     font=("Segoe UI", 12, "bold"),
                                     text_color=COLORS["success"])
            lbl_empty.pack(expand=True, pady=40)
            return

        self.btn_clear_all.configure(state="normal")

        # Sum total size
        total_bytes = sum(item["size_bytes"] for item in data_list)
        total_size_mb = total_bytes / (1024 * 1024)
        
        summary_lbl = ctk.CTkLabel(self.list_scroll, 
                                   text=f"📂 Found {len(data_list)} temporary experiments occupying {total_size_mb:.1f} MB on disk.",
                                   font=("Segoe UI", 11, "bold"),
                                   text_color=COLORS["text"],
                                   anchor="w")
        summary_lbl.pack(fill="x", padx=15, pady=(10, 15))

        # Render list of cards
        for idx, item in enumerate(data_list):
            card = ctk.CTkFrame(self.list_scroll, fg_color=COLORS["card"],
                                 border_color=COLORS["border"], border_width=1, corner_radius=10)
            card.pack(fill="x", padx=15, pady=5)
            card.grid_columnconfigure(0, weight=1)
            card.grid_columnconfigure(1, weight=0)

            # Details block
            details = ctk.CTkFrame(card, fg_color="transparent")
            details.grid(row=0, column=0, sticky="nsew", padx=15, pady=12)

            name_lbl = ctk.CTkLabel(details, text=item["name"], font=("Segoe UI", 13, "bold"),
                                    text_color=COLORS["text"], anchor="w")
            name_lbl.pack(anchor="w")

            # Status pill styling
            status = str(item["status"]).lower()
            status_text = status.upper()
            status_color = COLORS["warning"]
            if "completed" in status:
                status_color = COLORS["success"]
            elif "error" in status or "stopped" in status or "interrupted" in status:
                status_color = COLORS["error"]

            best_acc_val = item.get("best_acc", "—")
            if isinstance(best_acc_val, float):
                best_acc_text = f"{best_acc_val:.2%}"
            else:
                best_acc_text = str(best_acc_val)
            info_text = f"Status: {status_text}  •  Epoch: {item['epoch']}  •  Best Acc: {best_acc_text}  •  Size: {item['size_bytes'] / (1024*1024):.1f} MB  •  Modified: {item['modified']}"
            info_lbl = ctk.CTkLabel(details, text=info_text, font=("Segoe UI", 11),
                                    text_color=COLORS["text_dim"], anchor="w")
            info_lbl.pack(anchor="w", pady=(2, 0))

            # Action button block
            btn_delete = ctk.CTkButton(card, text="🗑 Delete", width=80, height=30,
                                       fg_color="transparent", border_color=COLORS["error"], border_width=1,
                                       text_color=COLORS["error"], hover_color="#c0392b", font=("Segoe UI", 11, "bold"),
                                       command=lambda data=item: self._delete_checkpoint_confirm(data))
            btn_delete.grid(row=0, column=1, padx=15, pady=12)

    def _delete_checkpoint_confirm(self, item):
        """Safely delete files associated with a single experiment."""
        for f in item["files"]:
            try:
                if f.exists():
                    f.unlink()
            except Exception as e:
                print(f"Failed to delete checkpoint file {f}: {e}")
                
        # Also clean up json state
        state_file = self.hcfg.CHECKPOINT_DIR / f"{item['name']}_training_state.json"
        try:
            if state_file.exists():
                state_file.unlink()
        except:
            pass
            
        self._refresh_unfinished_checkpoints()

    def _clear_all_unfinished(self):
        """Wipe out all temporary checkpoint files in one click."""
        data_list = self._find_unfinished_checkpoints()
        for item in data_list:
            for f in item["files"]:
                try:
                    if f.exists():
                        f.unlink()
                except Exception as e:
                    print(f"Failed to clear {f}: {e}")
            state_file = self.hcfg.CHECKPOINT_DIR / f"{item['name']}_training_state.json"
            try:
                if state_file.exists():
                    state_file.unlink()
            except:
                pass
                
        self._refresh_unfinished_checkpoints()

    def _start_metrics_generation(self):
        """Disables controls, shows progress bar, and starts background evaluation thread."""
        exp_name = self.exp_var.get()
        if exp_name == "None" or not exp_name:
            return
            
        self.btn_gen_metrics.configure(state="disabled")
        # Hide the trigger row, show progress bar frame
        self.banner_inner.pack_forget()
        self.progress_frame.pack(fill="x", padx=15, pady=10)
        self.gen_progress_bar.set(0.0)
        self.gen_progress_lbl.configure(text="Initializing model and weights...", text_color=COLORS["text_dim"])
        
        # Disable dropdown selection to prevent user switching mid-evaluation
        self.exp_menu.configure(state="disabled")
        
        # Spawn thread
        threading.Thread(target=self._generate_metrics_async, args=(exp_name,), daemon=True).start()

    def _generate_metrics_async(self, exp_name):
        import torch
        import torch.nn as nn
        from har.predict import load_model
        from har.dataset import create_cached_dataloaders, create_dataloaders
        from har.evaluate import compute_metrics
        import numpy as np

        def update_progress(msg, progress_val, color=None):
            if color is None:
                color = COLORS["text_dim"]
            self.after(0, lambda: (
                self.gen_progress_bar.set(progress_val),
                self.gen_progress_lbl.configure(text=msg, text_color=color)
            ))

        try:
            # 1. Locate checkpoint weights
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt_path = None
            
            # Check synonyms for checkpoint naming
            for suffix in ["_best.pth", "_pause.pth", "_latest.pth"]:
                p = self.hcfg.CHECKPOINT_DIR / f"{exp_name}{suffix}"
                if p.exists():
                    ckpt_path = p
                    break
                    
            if not ckpt_path:
                update_progress("❌ Error: Checkpoint file not found.", 0.0, COLORS["error"])
                self.after(3000, self._reset_gen_ui)
                return

            update_progress("🔄 Loading model checkpoint...", 0.1)
            
            # Load model and class names using backward compatible loader
            model, class_names = load_model(str(ckpt_path), device)
            model.eval()

            # 2. Parse config from training state JSON
            state_path = self.hcfg.CHECKPOINT_DIR / f"{exp_name}_training_state.json"
            if not state_path.exists():
                state_path = self.hcfg.CHECKPOINT_DIR / f"{exp_name}_state.json"
                
            if not state_path.exists():
                update_progress("❌ Error: Configuration state JSON not found.", 0.0, COLORS["error"])
                self.after(3000, self._reset_gen_ui)
                return
                
            with open(state_path) as fh:
                state_dict = json.load(fh)
                
            config_dict = state_dict.get("config", {})
            if not config_dict:
                update_progress("❌ Error: Invalid config inside state JSON.", 0.0, COLORS["error"])
                self.after(3000, self._reset_gen_ui)
                return

            # Recreate dataset dataloader
            use_cache = config_dict.get("use_cache", True)
            cache_path = config_dict.get("cache_path", "")
            dataset_path = config_dict.get("dataset_path", "")
            batch_size = config_dict.get("batch_size", 16)
            
            update_progress("📁 Setting up validation stream...", 0.2)
            
            # Validate cache or video dataset paths existence
            if use_cache and cache_path and Path(cache_path).exists():
                _, val_loader, loader_classes = create_cached_dataloaders(
                    cache_dir=Path(cache_path),
                    batch_size=batch_size,
                    num_workers=0, # thread-safe 0 workers
                )
            elif dataset_path and Path(dataset_path).exists():
                _, val_loader, loader_classes = create_dataloaders(
                    data_dir=Path(dataset_path),
                    batch_size=batch_size,
                    num_workers=0,
                )
            else:
                # Path missing on this workstation! Show path error
                update_progress("❌ Error: Dataset or Cache folder path is missing or invalid on this workstation.", 0.0, COLORS["error"])
                self.after(5000, self._reset_gen_ui)
                return

            if not class_names and loader_classes:
                class_names = loader_classes

            # 3. Batch prediction loop
            update_progress("⚡ Starting batch evaluation...", 0.3)
            
            all_preds, all_labels = [], []
            total_batches = len(val_loader)
            
            for idx, (videos, labels) in enumerate(val_loader):
                videos = videos.to(device, non_blocking=True)
                
                # Normalize frame values if Colab architecture
                if getattr(model, "is_colab", False):
                    videos = videos * 255.0

                with torch.no_grad():
                    if torch.cuda.is_available() and config_dict.get("use_amp", True):
                        with torch.amp.autocast('cuda', enabled=True):
                            outputs = model(videos)
                    else:
                        outputs = model(videos)
                        
                    probs = torch.softmax(outputs.float(), dim=1)
                    preds = probs.argmax(dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())
                
                # Scale progress between 0.3 and 0.9
                progress_val = 0.3 + 0.6 * ((idx + 1) / total_batches)
                update_progress(f"Evaluating: Batch {idx+1}/{total_batches} ({((idx+1)/total_batches):.0%})", progress_val)

            # 4. Compute metrics
            update_progress("📊 Compiling classification metrics...", 0.95)
            
            metrics = compute_metrics(np.array(all_preds), np.array(all_labels), class_names)
            
            # Save metrics JSON to METRICS_DIR
            self.hcfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
            metrics_path = self.hcfg.METRICS_DIR / f"{exp_name}_metrics.json"
            
            with open(metrics_path, "w") as fh:
                json.dump(metrics, fh, indent=2)
                
            update_progress("✅ Complete! Metrics generated successfully.", 1.0, COLORS["success"])
            
            # Reload results to instantly display curves and newly generated metrics
            self.after(1500, lambda: self._on_metrics_generation_complete(exp_name))

        except Exception as e:
            import traceback
            traceback.print_exc()
            update_progress(f"❌ Error: {str(e)}", 0.0, COLORS["error"])
            self.after(5000, self._reset_gen_ui)

    def _reset_gen_ui(self):
        """Restore GUI controls back to normal idle state."""
        self.exp_menu.configure(state="normal")
        self.progress_frame.pack_forget()
        self.banner_inner.pack(fill="x", padx=15, pady=10)
        self.btn_gen_metrics.configure(state="normal")

    def _on_metrics_generation_complete(self, exp_name):
        """Restore dropdown controls and reload Results plots."""
        self.exp_menu.configure(state="normal")
        # Hide the generation banner entirely since has_metrics is now True!
        self.metrics_gen_banner.grid_forget()
        
        # Refresh the overall experiment menu values (so that it updates if completed)
        exps = self._find_experiments()
        self.exp_menu.configure(values=exps or ["None"])
        self.exp_var.set(exp_name)
        
        # Load and draw results
        self._load_results(exp_name)

    def on_theme_changed(self, mode):
        self.configure(fg_color=COLORS["bg"])
        current = self.exp_var.get()
        if current and current != "None":
            self._load_results(current)
