"""Training Monitor — premium 3-zone layout with live analytics."""

import queue
import tkinter as tk
import customtkinter as ctk
import subprocess
import sys
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.theme import COLORS


class TrainingMonitorFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=COLORS["bg"], corner_radius=0)
        self.app = app
        self._log_queue = queue.Queue()
        self._auto_scroll = True
        self._max_log_lines = 500
        self._poll_id = None

        # 3-zone grid: header / [charts | right-panel] / status-bar
        self.grid_columnconfigure(0, weight=55)
        self.grid_columnconfigure(1, weight=45)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_charts()
        self._build_right_panel()
        self._build_status_bar()

        self.app.trainer_service.add_callback(self._on_progress)
        self._poll_console()

    # ─────────────────────────── HEADER ────────────────────────────
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=30, pady=(20, 10))

        ctk.CTkLabel(header, text="Training Monitor",
                     font=("Segoe UI", 24, "bold"),
                     text_color=COLORS["text"]).pack(side="left")

        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.pack(side="right")

        self.pause_btn = ctk.CTkButton(
            btn_frame, text="⏸  Pause", width=105, height=36,
            fg_color=COLORS["warning"], text_color=("#000", "#000"),
            font=("Segoe UI", 12, "bold"), corner_radius=8,
            command=self._pause)
        self.pause_btn.pack(side="left", padx=4)

        self.resume_btn = ctk.CTkButton(
            btn_frame, text="▶  Resume", width=105, height=36,
            fg_color=COLORS["success"], font=("Segoe UI", 12, "bold"),
            corner_radius=8, command=self._resume)
        self.resume_btn.pack(side="left", padx=4)

        self.stop_btn = ctk.CTkButton(
            btn_frame, text="⏹  Stop", width=105, height=36,
            fg_color=COLORS["error"], font=("Segoe UI", 12, "bold"),
            corner_radius=8, command=self._stop)
        self.stop_btn.pack(side="left", padx=4)

        self.tb_btn = ctk.CTkButton(
            btn_frame, text="TensorBoard", width=120, height=36,
            fg_color=COLORS["border"], text_color=COLORS["text"],
            font=("Segoe UI", 12), corner_radius=8,
            command=self._launch_tensorboard)
        self.tb_btn.pack(side="left", padx=(14, 0))

    # ──────────────────────────── CHARTS ───────────────────────────
    def _build_charts(self):
        card = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=14,
                            border_width=1, border_color=COLORS["card_border"])
        card.grid(row=1, column=0, sticky="nsew", padx=(30, 8), pady=5)

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

        self.canvas = FigureCanvasTkAgg(self.fig, master=card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    def _apply_chart_titles(self, text_col):
        for ax, title in (
            (self.ax_loss,  "Loss (Log Scale)"),
            (self.ax_acc,   "Accuracy (%)"),
            (self.ax_lr,    "Learning Rate"),
            (self.ax_speed, "Epoch Duration (s)"),
        ):
            ax.set_title(title, color=text_col, fontsize=10, fontweight="bold", pad=6)

    # ─────────────────────── RIGHT PANEL ───────────────────────────
    def _build_right_panel(self):
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 30), pady=5)
        right.grid_rowconfigure(0, weight=0)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._build_stat_cards(right)
        self._build_console(right)

    def _build_stat_cards(self, parent):
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        container.grid_columnconfigure((0, 1), weight=1)

        CARDS = [
            ("card_loss",  "Train Loss",       "—", "#6c5ce7"),
            ("card_acc",   "Val Accuracy",     "—", "#00cec9"),
            ("card_best",  "Best Accuracy ★",  "—", "#ffd700"),
            ("card_speed", "Epoch Speed",      "—", "#00b894"),
        ]
        for i, (attr, title, init, color) in enumerate(CARDS):
            r, c = divmod(i, 2)
            card = ctk.CTkFrame(container, fg_color=COLORS["card"],
                                corner_radius=12, border_width=1,
                                border_color=COLORS["card_border"])
            card.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")
            container.grid_rowconfigure(r, weight=1)

            # Colored top accent strip
            ctk.CTkFrame(card, fg_color=color, height=3,
                         corner_radius=0).pack(fill="x")

            ctk.CTkLabel(card, text=title, font=("Segoe UI", 9),
                         text_color=COLORS["text_dim"]).pack(
                anchor="w", padx=10, pady=(6, 0))

            lbl = ctk.CTkLabel(card, text=init,
                               font=("Segoe UI", 18, "bold"),
                               text_color=color)
            lbl.pack(anchor="w", padx=10, pady=(1, 8))
            setattr(self, attr, lbl)

    def _build_console(self, parent):
        card = ctk.CTkFrame(parent, fg_color=COLORS["card"], corner_radius=12,
                            border_width=1, border_color=COLORS["card_border"])
        card.grid(row=1, column=0, sticky="nsew")
        card.grid_rowconfigure(1, weight=1)
        card.grid_columnconfigure(0, weight=1)

        # Header row
        hdr = ctk.CTkFrame(card, fg_color="transparent")
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 4))

        ctk.CTkLabel(hdr, text="● Live Output Console",
                     font=("Segoe UI", 11, "bold"),
                     text_color=COLORS["text"]).pack(side="left")

        ctk.CTkButton(hdr, text="Clear", width=52, height=24,
                      font=("Segoe UI", 10), corner_radius=6,
                      fg_color=COLORS["border"], text_color=COLORS["text_dim"],
                      hover_color=COLORS["input_bg"],
                      command=self._clear_console).pack(side="right")

        # tk.Text for tag-based coloring
        idx = 1 if ctk.get_appearance_mode().lower() == "dark" else 0
        bg = COLORS["console_bg"][idx]

        self._console = tk.Text(
            card, bg=bg, fg="#a0a8b8",
            font=("Consolas", 10), wrap="word",
            relief="flat", bd=0,
            insertbackground="#6c5ce7",
            selectbackground="#252836",
        )
        self._console.grid(row=1, column=0, sticky="nsew",
                           padx=(12, 0), pady=(0, 12))

        sb = ctk.CTkScrollbar(card, command=self._console.yview)
        sb.grid(row=1, column=1, sticky="ns", padx=(0, 6), pady=(0, 12))
        self._console.configure(yscrollcommand=sb.set)
        self._console.configure(state="disabled")

        # Color tags
        self._console.tag_config("error",   foreground="#ff6b6b")
        self._console.tag_config("warning", foreground="#fdcb6e")
        self._console.tag_config("epoch",   foreground="#00b894")
        self._console.tag_config("best",    foreground="#ffd700")
        self._console.tag_config("info",    foreground="#74b9ff")
        self._console.tag_config("default", foreground="#a0a8b8")

        self._console.bind("<MouseWheel>", self._on_console_scroll)
        self._console.bind("<Button-4>",   self._on_console_scroll)
        self._console.bind("<Button-5>",   self._on_console_scroll)

        # Start redirector
        from gui.services.console_redirector import ConsoleRedirector
        self.redirector = ConsoleRedirector(self._log_queue)
        self.redirector.start()

    # ─────────────────── CONSOLE HELPERS ───────────────────────────
    def _on_console_scroll(self, _event):
        self.after(50, self._check_scroll_pos)

    def _check_scroll_pos(self):
        try:
            self._auto_scroll = self._console.yview()[1] >= 0.98
        except Exception:
            pass

    def _poll_console(self):
        """Drain the queue and write to console — always on main thread."""
        chunks = []
        try:
            while True:
                chunks.append(self._log_queue.get_nowait())
        except queue.Empty:
            pass

        if chunks:
            self._append_to_console("".join(chunks))

        self._poll_id = self.after(120, self._poll_console)

    def _classify(self, line: str) -> str:
        lo = line.lower()
        if any(k in lo for k in ("error", "exception", "traceback", "failed")):
            return "error"
        if any(k in lo for k in ("warning", "warn", "deprecated")):
            return "warning"
        if "best" in lo or "* best" in lo:
            return "best"
        if any(k in lo for k in ("epoch", "val_acc", "train_acc")):
            return "epoch"
        if any(k in lo for k in ("loading", "building", "model", "dataset", "cached", "split")):
            return "info"
        return "default"

    def _append_to_console(self, text: str):
        self._console.configure(state="normal")
        for line in text.splitlines(keepends=True):
            self._console.insert("end", line, self._classify(line))

        # Trim excess lines
        line_count = int(self._console.index("end-1c").split(".")[0])
        if line_count > self._max_log_lines:
            self._console.delete("1.0", f"{line_count - self._max_log_lines + 1}.0")

        self._console.configure(state="disabled")
        if self._auto_scroll:
            self._console.see("end")

    def _clear_console(self):
        self._console.configure(state="normal")
        self._console.delete("1.0", "end")
        self._console.configure(state="disabled")

    # ─────────────────────── STATUS BAR ────────────────────────────
    def _build_status_bar(self):
        bar = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=12,
                           border_width=1, border_color=COLORS["card_border"])
        bar.grid(row=2, column=0, columnspan=2, sticky="ew",
                 padx=30, pady=(5, 20))
        bar.grid_columnconfigure(tuple(range(6)), weight=1)

        # Metric row
        metrics = ctk.CTkFrame(bar, fg_color="transparent")
        metrics.grid(row=0, column=0, columnspan=6, sticky="ew",
                     padx=15, pady=(10, 4))
        metrics.grid_columnconfigure(tuple(range(6)), weight=1)

        self.status_pill = ctk.CTkLabel(
            metrics, text="  ● IDLE  ",
            font=("Segoe UI", 12, "bold"),
            fg_color=COLORS["border"], text_color=COLORS["text_dim"],
            corner_radius=8)
        self.status_pill.grid(row=0, column=0, sticky="w", padx=(0, 10))

        for i, (attr, text) in enumerate([
            ("epoch_lbl", "Epoch: -/-"),
            ("batch_lbl", "Batch: -/-"),
            ("best_lbl",  "Best: -"),
            ("eta_lbl",   "ETA: -"),
        ], start=1):
            lbl = ctk.CTkLabel(metrics, text=text,
                               font=("Segoe UI", 12),
                               text_color=COLORS["text"])
            lbl.grid(row=0, column=i, padx=8)
            setattr(self, attr, lbl)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            bar, progress_color=COLORS["accent"],
            fg_color=COLORS["progress_bg"], height=5, corner_radius=3)
        self.progress_bar.grid(row=1, column=0, columnspan=6,
                               sticky="ew", padx=15, pady=(0, 5))
        self.progress_bar.set(0)

        # Message
        self.msg_lbl = ctk.CTkLabel(
            bar, text="", font=("Segoe UI", 10),
            text_color=COLORS["text_dim"], anchor="w")
        self.msg_lbl.grid(row=2, column=0, columnspan=6,
                          sticky="w", padx=15, pady=(0, 10))

    # ──────────────────── PROGRESS CALLBACK ────────────────────────
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

        self.status_pill.configure(
            text=f"  ● {s.upper()}  ", text_color=fg, fg_color=bg)

        self.epoch_lbl.configure(
            text=f"Epoch: {progress.current_epoch}/{progress.total_epochs}")

        if progress.total_batches > 0 and s not in ["paused", "stopped", "completed", "idle"]:
            is_val = "validating" in progress.message.lower()
            prefix = "Val Batch" if is_val else "Batch"
            self.batch_lbl.configure(
                text=f"{prefix}: {progress.current_batch}/{progress.total_batches}")
        else:
            self.batch_lbl.configure(text="Batch: -/-")

        self.best_lbl.configure(
            text=f"Best: {progress.best_val_acc:.2%}")

        mins = int(progress.eta_seconds // 60)
        self.eta_lbl.configure(text=f"ETA: {mins // 60}h {mins % 60}m")


        if progress.total_epochs > 0:
            if progress.total_batches > 0 and progress.current_batch > 0 and s not in ["paused", "stopped", "completed", "idle"]:
                is_val = "validating" in progress.message.lower()
                batch_fraction = (progress.current_batch - 1) / progress.total_batches
                batch_fraction = max(0.0, min(1.0, batch_fraction))
                
                epoch_base = (progress.current_epoch - 1) / progress.total_epochs
                epoch_segment = 1.0 / progress.total_epochs
                
                if is_val:
                    # Validation phase: occupies the final 15% of the epoch segment
                    smooth_p = epoch_base + epoch_segment * (0.85 + 0.15 * batch_fraction)
                else:
                    # Training phase: occupies the first 85% of the epoch segment
                    smooth_p = epoch_base + epoch_segment * (0.85 * batch_fraction)
                    
                self.progress_bar.set(max(0.0, min(1.0, smooth_p)))
            else:
                self.progress_bar.set(progress.current_epoch / progress.total_epochs)

        self.msg_lbl.configure(text=progress.message)

        # Stat cards
        self.card_loss.configure(
            text=f"{progress.train_loss:.4f}" if progress.train_loss else "—")
        self.card_acc.configure(
            text=f"{progress.val_acc:.2%}" if progress.val_acc else "—")
        self.card_best.configure(
            text=f"{progress.best_val_acc:.2%}" if progress.best_val_acc else "—")

        h = progress.history
        if h.get("epoch_time"):
            avg = sum(h["epoch_time"]) / len(h["epoch_time"])
            self.card_speed.configure(text=f"{avg / 60:.1f} min/ep")

        # Smart chart drawing: handle resets and redraws when history length changes
        current_last_drawn = getattr(self, "_last_drawn_epoch", 0)
        num_points = len(h.get("train_loss", []))

        if num_points == 0:
            if current_last_drawn > 0:
                self._clear_charts()
                self._last_drawn_epoch = 0
        else:
            if current_last_drawn != num_points:
                self._redraw_charts(progress)
                self._last_drawn_epoch = num_points

    # ────────────────────── CHART DRAWING ──────────────────────────
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


    # ────────────────────── CHART DRAWING ──────────────────────────
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
        C_BEST  = "#ffd700"

        n = len(h["train_loss"])
        epochs = list(range(1, n + 1))

        best_epoch = None
        if h.get("val_acc"):
            best_epoch = h["val_acc"].index(max(h["val_acc"])) + 1

        for ax in (self.ax_loss, self.ax_acc, self.ax_lr, self.ax_speed):
            ax.clear()
            ax.set_facecolor(plot_bg)
            ax.tick_params(colors=text_dim, labelsize=8)
            ax.spines["bottom"].set_color(plot_grid)
            ax.spines["left"].set_color(plot_grid)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color=plot_grid, linewidth=0.5, alpha=0.6)
            if best_epoch:
                ax.axvline(best_epoch, color=C_BEST, linestyle="--",
                           linewidth=1.2, alpha=0.7, zorder=0)

        # Loss (log)
        tl, vl = h["train_loss"], h["val_loss"]
        self.ax_loss.plot(epochs, tl, color=C_TRAIN, linewidth=1.8, label="Train")
        self.ax_loss.plot(epochs, vl, color=C_VAL,   linewidth=1.8, label="Val")
        self.ax_loss.fill_between(epochs, tl, alpha=0.09, color=C_TRAIN)
        self.ax_loss.fill_between(epochs, vl, alpha=0.09, color=C_VAL)
        self.ax_loss.set_yscale("log")
        self.ax_loss.legend(facecolor=plot_bg, edgecolor=plot_grid,
                            labelcolor=text_col, fontsize=8)

        # Accuracy (%)
        ta = [v * 100 for v in h["train_acc"]]
        va = [v * 100 for v in h["val_acc"]]
        self.ax_acc.plot(epochs, ta, color=C_TRAIN, linewidth=1.8, label="Train")
        self.ax_acc.plot(epochs, va, color=C_VAL,   linewidth=1.8, label="Val")
        self.ax_acc.fill_between(epochs, ta, alpha=0.09, color=C_TRAIN)
        self.ax_acc.fill_between(epochs, va, alpha=0.09, color=C_VAL)
        self.ax_acc.set_ylabel("%", color=text_dim, fontsize=8)
        self.ax_acc.legend(facecolor=plot_bg, edgecolor=plot_grid,
                           labelcolor=text_col, fontsize=8)

        # Learning Rate
        if h.get("lr"):
            lrs = h["lr"][:n]
            ep2 = epochs[:len(lrs)]
            self.ax_lr.plot(ep2, lrs, color=C_LR, linewidth=1.8)
            self.ax_lr.fill_between(ep2, lrs, alpha=0.10, color=C_LR)

        # Epoch speed
        if h.get("epoch_time"):
            times = h["epoch_time"][:n]
            ep3   = epochs[:len(times)]
            self.ax_speed.plot(ep3, times, color=C_SPEED, linewidth=1.8)
            self.ax_speed.fill_between(ep3, times, alpha=0.10, color=C_SPEED)

        self._apply_chart_titles(text_col)
        self.canvas.draw_idle()

    # ─────────────────────── CONTROLS ──────────────────────────────
    def _pause(self):
        self.app.trainer_service.pause()

    def _resume(self):
        if self.app.trainer_service.progress.status == "idle":
            self._show_resume_manager()
        else:
            self.app.trainer_service.resume()

    def _show_resume_manager(self):
        from gui.services.trainer_service import find_paused_experiments
        paused = find_paused_experiments()
        if not paused:
            self.msg_lbl.configure(
                text="No paused or interrupted sessions found.",
                text_color=COLORS["warning"])
            return

        dlg = ctk.CTkToplevel(self)
        dlg.title("Resume Manager")
        dlg.geometry("500x300")
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()
        dlg.update_idletasks()
        x = (dlg.winfo_screenwidth()  - 500) // 2
        y = (dlg.winfo_screenheight() - 300) // 2
        dlg.geometry(f"+{x}+{y}")

        ctk.CTkLabel(dlg, text="Select Session to Resume",
                     font=("Segoe UI", 18, "bold"),
                     text_color=COLORS["accent"]).pack(pady=(20, 10))

        options, state_map = [], {}
        for state in paused:
            exp   = state.get("config", {}).get("experiment_name", "Unknown")
            epoch = state.get("current_epoch", 0)
            st    = state.get("status", "unknown")
            lbl   = f"{exp} (Epoch {epoch}) [{st.upper()}]"
            options.append(lbl)
            state_map[lbl] = state

        var = ctk.StringVar(value=options[-1])
        ctk.CTkOptionMenu(dlg, variable=var, values=options, width=420,
                          font=("Segoe UI", 12), fg_color=COLORS["input_bg"],
                          button_color=COLORS["accent"]).pack(pady=20)

        bf = ctk.CTkFrame(dlg, fg_color="transparent")
        bf.pack(pady=10)

        def on_resume():
            st = state_map.get(var.get())
            if st:
                self.app.trainer_service.load_paused_state(st)
                self.app.trainer_service.resume()
            dlg.destroy()

        ctk.CTkButton(bf, text="Cancel", width=110,
                      fg_color=COLORS["border"], text_color=COLORS["text"],
                      command=dlg.destroy).pack(side="left", padx=8)
        ctk.CTkButton(bf, text="Resume Selected", width=150,
                      fg_color=COLORS["success"], text_color="#000000",
                      command=on_resume).pack(side="left", padx=8)

    def _stop(self):
        self.app.trainer_service.stop()

    def _launch_tensorboard(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from har import config as hcfg
        try:
            subprocess.Popen(
                [sys.executable, "-c", "from tensorboard import main; main.run_main()",
                 "--logdir", str(hcfg.TENSORBOARD_DIR), "--port", "6006"],
                creationflags=subprocess.CREATE_NO_WINDOW
                if sys.platform == "win32" else 0,
            )
            import webbrowser
            self.after(2000, lambda: webbrowser.open("http://localhost:6006"))
        except Exception as e:
            self.msg_lbl.configure(text=f"TensorBoard error: {e}")



    def on_theme_changed(self, mode):
        self.configure(fg_color=COLORS["bg"])
        idx = 1 if mode.lower() == "dark" else 0
        
        # 1. Update standard console bg
        self._console.configure(bg=COLORS["console_bg"][idx])
        
        # 2. Update syntax tag colors for high contrast in current mode
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
        h = self.app.trainer_service.progress.history
        if h.get("train_loss"):
            self._redraw_charts(self.app.trainer_service.progress)
        else:
            self._clear_charts()

    # ──────────────────────── CLEANUP ──────────────────────────────
    def destroy(self):
        if self._poll_id:
            self.after_cancel(self._poll_id)
        if hasattr(self, "redirector"):
            self.redirector.stop()
        self.app.trainer_service.stop()
        super().destroy()
