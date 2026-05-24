"""SOTA Benchmark Page — Compare our model against pretrained SOTA transformers."""

import customtkinter as ctk
from tkinter import filedialog
from pathlib import Path
import os
import threading
import sys
import time

from gui.theme import COLORS


class SOTABenchmarkFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=COLORS["bg"], corner_radius=0)
        self.app = app
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._cancel_flag = False
        self._benchmark_running = False
        self._report_text = ""

        # ── Title ──
        ctk.CTkLabel(self, text="SOTA Benchmark", font=("Segoe UI", 24, "bold"),
                     text_color=COLORS["text"]).grid(row=0, column=0, columnspan=2,
                                                      sticky="w", padx=30, pady=(25, 5))
        ctk.CTkLabel(self, text="Compare your model against pretrained state-of-the-art video classifiers",
                     font=("Segoe UI", 11), text_color=COLORS["text_dim"]).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=30, pady=(50, 15))

        # ═══════════════════════════════════════════
        # LEFT PANEL — Controls
        # ═══════════════════════════════════════════
        left = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=12,
                            border_width=1, border_color=COLORS["border"])
        left.grid(row=1, column=0, sticky="nsew", padx=(30, 10), pady=(0, 20))

        # ── Model Selector ──
        ctk.CTkLabel(left, text="Your Model", font=("Segoe UI", 13, "bold"),
                     text_color=COLORS["accent"]).pack(anchor="w", padx=20, pady=(18, 5))

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from har import config as hcfg
        self.hcfg = hcfg

        checkpoints = [p.name for p in hcfg.CHECKPOINT_DIR.glob("*_best.pth")]
        self.model_var = ctk.StringVar(value=checkpoints[0] if checkpoints else "No models")
        ctk.CTkOptionMenu(left, variable=self.model_var, values=checkpoints or ["No models"],
                          font=("Segoe UI", 11), fg_color=COLORS["input_bg"],
                          button_color=COLORS["accent"]).pack(fill="x", padx=20, pady=5)

        # ── Separator ──
        ctk.CTkFrame(left, height=1, fg_color=COLORS["border"]).pack(fill="x", padx=20, pady=12)

        # ── Video Source ──
        ctk.CTkLabel(left, text="Video Source", font=("Segoe UI", 13, "bold"),
                     text_color=COLORS["accent"]).pack(anchor="w", padx=20, pady=(0, 5))

        self.source_var = ctk.StringVar(value="local")
        self._src_frame = ctk.CTkFrame(left, fg_color="transparent")
        self._src_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkRadioButton(self._src_frame, text="Local Video File", variable=self.source_var,
                           value="local", font=("Segoe UI", 11), text_color=COLORS["text"],
                           fg_color=COLORS["accent"], hover_color=COLORS["accent"],
                           command=self._toggle_source).pack(anchor="w", pady=2)
        ctk.CTkRadioButton(self._src_frame, text="YouTube URL", variable=self.source_var,
                           value="youtube", font=("Segoe UI", 11), text_color=COLORS["text"],
                           fg_color=COLORS["accent"], hover_color=COLORS["accent"],
                           command=self._toggle_source).pack(anchor="w", pady=2)

        # Local file controls
        self.local_frame = ctk.CTkFrame(left, fg_color="transparent")
        self.local_frame.pack(fill="x", padx=20, pady=5)

        self.file_path_var = ctk.StringVar(value="No file selected")
        ctk.CTkLabel(self.local_frame, textvariable=self.file_path_var, font=("Segoe UI", 10),
                     text_color=COLORS["text_dim"], wraplength=260, anchor="w",
                     justify="left").pack(fill="x", pady=2)
        ctk.CTkButton(self.local_frame, text="Browse Video File", font=("Segoe UI", 11),
                      fg_color=COLORS["input_bg"], hover_color=COLORS["border"],
                      border_width=1, border_color=COLORS["border"],
                      text_color=COLORS["text"], corner_radius=8, height=30,
                      command=self._browse_video).pack(fill="x", pady=3)

        # YouTube URL controls
        self.yt_frame = ctk.CTkFrame(left, fg_color="transparent")
        self.yt_url_var = ctk.StringVar(value="")
        ctk.CTkEntry(self.yt_frame, textvariable=self.yt_url_var, font=("Segoe UI", 10),
                     fg_color=COLORS["input_bg"], border_color=COLORS["border"],
                     height=32, placeholder_text="Paste YouTube URL here...").pack(fill="x", pady=3)

        self._toggle_source()

        # ── Separator ──
        ctk.CTkFrame(left, height=1, fg_color=COLORS["border"]).pack(fill="x", padx=20, pady=12)

        # ── SOTA Models Selection ──
        ctk.CTkLabel(left, text="SOTA Models to Benchmark", font=("Segoe UI", 13, "bold"),
                     text_color=COLORS["accent"]).pack(anchor="w", padx=20, pady=(0, 5))

        self.model_toggles = {}
        self.download_btns = {}
        self._model_rows = {}
        from har.sota_benchmark import SOTA_MODELS
        for name, info in SOTA_MODELS.items():
            var = ctk.BooleanVar(value=True)
            self.model_toggles[name] = var
            row = ctk.CTkFrame(left, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=2)
            self._model_rows[name] = row

            ctk.CTkCheckBox(row, text=f"{name}", variable=var,
                            font=("Segoe UI", 11), text_color=COLORS["text"],
                            fg_color=COLORS["accent"], hover_color=COLORS["accent"],
                            checkbox_width=18, checkbox_height=18).pack(side="left")

            # Source badge + params
            src = info.get("source", "huggingface")
            badge = "🤗" if src == "huggingface" else "🔥"
            ctk.CTkLabel(row, text=f"  {badge} {info['params']} · {info['year']}",
                         font=("Segoe UI", 9), text_color=COLORS["text_dim"]).pack(side="left")

            # Download button (starts as "…" = checking)
            dl_btn = ctk.CTkButton(
                row, text="…", width=28, height=22, corner_radius=6,
                font=("Segoe UI", 11), fg_color=COLORS["input_bg"],
                hover_color=COLORS["border"], border_width=1,
                border_color=COLORS["border"], text_color=COLORS["text_dim"],
                state="disabled",
                command=lambda n=name, i=info: self._download_model(n, i))
            dl_btn.pack(side="right", padx=(4, 0))
            self.download_btns[name] = dl_btn

        # Check cache status asynchronously (don't block GUI startup)
        self.after(500, self._check_cache_status)

        # ── Separator ──
        ctk.CTkFrame(left, height=1, fg_color=COLORS["border"]).pack(fill="x", padx=20, pady=12)

        # ── Action Buttons ──
        self.start_btn = ctk.CTkButton(
            left, text="▶  Start SOTA Benchmark", font=("Segoe UI", 14, "bold"),
            fg_color=COLORS["success"], hover_color=COLORS["success"],
            corner_radius=10, height=44, command=self._start_benchmark)
        self.start_btn.pack(padx=20, pady=(5, 5), fill="x")

        self.cancel_btn = ctk.CTkButton(
            left, text="Cancel", font=("Segoe UI", 11),
            fg_color=COLORS["error"], hover_color=COLORS["error"],
            corner_radius=8, height=30, command=self._cancel_benchmark, state="disabled")
        self.cancel_btn.pack(padx=20, pady=(0, 10), fill="x")

        # ── Progress Section ──
        self.progress_frame = ctk.CTkFrame(left, fg_color=COLORS["bg"], corner_radius=8)
        self.progress_frame.pack(fill="x", padx=20, pady=(0, 15))

        self.progress_label = ctk.CTkLabel(self.progress_frame, text="Ready",
                                            font=("Segoe UI", 10),
                                            text_color=COLORS["text_dim"])
        self.progress_label.pack(anchor="w", padx=10, pady=(8, 2))

        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame, width=200, height=10,
            progress_color=COLORS["accent"], fg_color=COLORS["progress_bg"])
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 4))
        self.progress_bar.set(0)

        self.progress_detail = ctk.CTkLabel(self.progress_frame, text="",
                                             font=("Segoe UI", 9),
                                             text_color=COLORS["text_dim"])
        self.progress_detail.pack(anchor="w", padx=10, pady=(0, 8))

        # ═══════════════════════════════════════════
        # RIGHT PANEL — Results
        # ═══════════════════════════════════════════
        right = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=12,
                             border_width=1, border_color=COLORS["border"])
        right.grid(row=1, column=1, sticky="nsew", padx=(10, 30), pady=(0, 20))
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # Header with export buttons
        header = ctk.CTkFrame(right, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 5))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header, text="Benchmark Results", font=("Segoe UI", 14, "bold"),
                     text_color=COLORS["accent"]).grid(row=0, column=0, sticky="w")

        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.grid(row=0, column=1, sticky="e")

        self.export_md_btn = ctk.CTkButton(
            btn_frame, text="Export .md", width=80, height=26,
            font=("Segoe UI", 10), fg_color=COLORS["input_bg"],
            hover_color=COLORS["border"], border_width=1, border_color=COLORS["border"],
            text_color=COLORS["text"], corner_radius=6, state="disabled",
            command=lambda: self._export_report("md"))
        self.export_md_btn.pack(side="left", padx=3)

        self.export_pdf_btn = ctk.CTkButton(
            btn_frame, text="Export .pdf", width=80, height=26,
            font=("Segoe UI", 10), fg_color=COLORS["input_bg"],
            hover_color=COLORS["border"], border_width=1, border_color=COLORS["border"],
            text_color=COLORS["text"], corner_radius=6, state="disabled",
            command=lambda: self._export_report("pdf"))
        self.export_pdf_btn.pack(side="left", padx=3)

        # Results area — scrollable
        self.results_scroll = ctk.CTkScrollableFrame(
            right, fg_color="transparent", corner_radius=0)
        self.results_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.results_scroll.grid_columnconfigure(0, weight=1)

        # Placeholder
        self.placeholder = ctk.CTkLabel(
            self.results_scroll,
            text="Select a video and model, then click\n'Start SOTA Benchmark' to begin.\n\n"
                 "The benchmark will run each SOTA model\n"
                 "sequentially to avoid VRAM overflow.\n\n"
                 "Results will appear here with a full\ncomparison table and analysis.",
            font=("Segoe UI", 12), text_color=COLORS["text_dim"],
            justify="center")
        self.placeholder.pack(pady=80)

    # ──────────────────────────────────────────────
    # Model Cache Management
    # ──────────────────────────────────────────────
    def _check_cache_status(self):
        """Check which SOTA models are already cached (runs in background)."""
        def check():
            from har.sota_benchmark import SOTA_MODELS, is_model_cached

            for name, info in SOTA_MODELS.items():
                if name not in self.download_btns:
                    continue
                cached = is_model_cached(info)
                if cached:
                    # Hide the download button entirely when cached
                    self.after(0, lambda n=name: self.download_btns[n].pack_forget())
                else:
                    # Show ⬇ download button
                    self.after(0, lambda n=name: self.download_btns[n].configure(
                        text="⬇", state="normal",
                        fg_color=COLORS["accent"],
                        text_color=("#ffffff", "#ffffff"),
                        border_color=COLORS["accent"]))
        threading.Thread(target=check, daemon=True).start()

    def _download_model(self, name, info):
        """Download a single SOTA model in background with progress animation."""
        btn = self.download_btns[name]
        btn.configure(text="⏳", state="disabled",
                      fg_color=COLORS["warning"],
                      text_color=("#000000", "#000000"),
                      border_color=COLORS["warning"])

        # Start pulsing animation on progress bar
        self._dl_active = True
        self._dl_direction = 1
        self.progress_bar.set(0.05)
        self._set_progress(f"⬇ Downloading {name} ({info['params']})...", COLORS["warning"])
        self.progress_detail.configure(text=f"Source: {info.get('source', 'huggingface')}")
        self._pulse_download()

        def do_download():
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            try:
                from har.sota_benchmark import download_sota_model
                download_sota_model(info)
                self._dl_active = False
                self.after(0, lambda: (
                    btn.pack_forget(),
                    self.progress_bar.set(1.0),
                    self._set_progress(f"✓ {name} downloaded and cached", COLORS["success"]),
                    self.progress_detail.configure(text="Ready for benchmark"),
                ))
                # Reset progress bar after a short delay
                self.after(2000, lambda: (
                    self.progress_bar.set(0),
                    self._set_progress("Ready", COLORS["text_dim"]),
                    self.progress_detail.configure(text=""),
                ) if not self._benchmark_running else None)
            except Exception as e:
                err_msg = str(e)[:80]  # Truncate for display
                self._dl_active = False
                self.after(0, lambda: (
                    btn.configure(text="✗", state="normal",
                                  fg_color=COLORS["error"],
                                  text_color=("#ffffff", "#ffffff"),
                                  border_color=COLORS["error"]),
                    self.progress_bar.set(0),
                    self._set_progress(f"✗ {name} failed: {err_msg}", COLORS["error"]),
                    self.progress_detail.configure(text="Click ✗ to retry"),
                ))
        threading.Thread(target=do_download, daemon=True).start()

    def _pulse_download(self):
        """Animate progress bar with a bouncing pulse during downloads."""
        if not self._dl_active:
            return
        val = self.progress_bar.get()
        val += 0.03 * self._dl_direction
        if val >= 0.95:
            self._dl_direction = -1
        elif val <= 0.05:
            self._dl_direction = 1
        self.progress_bar.set(val)
        self.after(80, self._pulse_download)

    # ──────────────────────────────────────────────
    # Source Toggle
    # ──────────────────────────────────────────────
    def _toggle_source(self):
        if self.source_var.get() == "local":
            self.yt_frame.pack_forget()
            if not self.local_frame.winfo_ismapped():
                self.local_frame.pack(fill="x", padx=20, pady=5, after=self._src_frame)
        else:
            self.local_frame.pack_forget()
            if not self.yt_frame.winfo_ismapped():
                self.yt_frame.pack(fill="x", padx=20, pady=5, after=self._src_frame)

    def _browse_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")])
        if path:
            self.file_path_var.set(path)

    # ──────────────────────────────────────────────
    # Benchmark Execution
    # ──────────────────────────────────────────────
    def _start_benchmark(self):
        if self._benchmark_running:
            return

        model_name = self.model_var.get()
        if "No" in model_name:
            self._set_progress("Select a model checkpoint first.", COLORS["error"])
            return

        # Validate video source
        if self.source_var.get() == "local":
            video_path = self.file_path_var.get()
            if "No file" in video_path or not Path(video_path).exists():
                self._set_progress("Select a valid video file first.", COLORS["error"])
                return
            source_display = Path(video_path).name
        else:
            yt_url = self.yt_url_var.get().strip()
            if not yt_url or "youtube" not in yt_url.lower() and "youtu.be" not in yt_url.lower():
                self._set_progress("Enter a valid YouTube URL.", COLORS["error"])
                return
            source_display = yt_url
            video_path = None

        # Check which SOTA models are enabled
        enabled = [name for name, var in self.model_toggles.items() if var.get()]
        if not enabled:
            self._set_progress("Enable at least one SOTA model.", COLORS["error"])
            return

        # Lock UI
        self._benchmark_running = True
        self._cancel_flag = False
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.export_md_btn.configure(state="disabled")
        self.export_pdf_btn.configure(state="disabled")

        # Clear results
        for w in self.results_scroll.winfo_children():
            w.destroy()

        self._set_progress("Initialising benchmark...", COLORS["warning"])
        self.progress_bar.set(0)

        def run():
            try:
                import torch
                from har.sota_benchmark import (
                    extract_video_frames, download_youtube_video,
                    run_our_model, run_sota_benchmark, generate_benchmark_report,
                    SOTA_MODELS, check_transformers_available,
                )

                device = "cuda" if torch.cuda.is_available() else "cpu"

                # ── Step 1: Check dependencies ──
                if not check_transformers_available():
                    self.after(0, lambda: self._set_progress(
                        "Missing 'transformers'. Run: pip install transformers",
                        COLORS["error"]))
                    self.after(0, self._unlock_ui)
                    return

                # ── Step 2: Get video file path ──
                self.after(0, lambda: self._set_progress("Preparing video...", COLORS["warning"]))
                self.after(0, lambda: self.progress_bar.set(0.05))

                yt_tmp_path = None  # Track temp file for cleanup
                if self.source_var.get() == "youtube":
                    yt_url = self.yt_url_var.get().strip()
                    self.after(0, lambda: self._set_progress(
                        "Downloading YouTube video (this may take a moment)...", COLORS["warning"]))
                    yt_tmp_path = download_youtube_video(yt_url)
                    video_for_our = yt_tmp_path
                else:
                    video_for_our = video_path

                # Extract 16 frames for SOTA models from the SAME video file
                frames = extract_video_frames(video_for_our, n_frames=32)

                if self._cancel_flag:
                    self.after(0, lambda: self._set_progress("Benchmark cancelled.", COLORS["text_dim"]))
                    self.after(0, self._unlock_ui)
                    if yt_tmp_path:
                        try: Path(yt_tmp_path).unlink()
                        except OSError: pass
                    return

                # ── Step 3: Run our model on the FULL video ──
                self.after(0, lambda: self._set_progress("Running our R(2+1)D model...", COLORS["warning"]))
                self.after(0, lambda: self.progress_bar.set(0.15))

                ckpt_path = str(self.hcfg.CHECKPOINT_DIR / model_name)
                our_result = run_our_model(video_for_our, ckpt_path, device)

                # Clean up YouTube temp file after our model is done
                if yt_tmp_path:
                    try: Path(yt_tmp_path).unlink()
                    except OSError: pass

                if self._cancel_flag:
                    self.after(0, lambda: self._set_progress("Benchmark cancelled.", COLORS["text_dim"]))
                    self.after(0, self._unlock_ui)
                    return

                # Show our model result immediately
                self.after(0, lambda: self._add_our_result(our_result))

                # ── Step 4: Run SOTA models ──
                total_sota = len(enabled)

                def on_progress(name, step, total, msg):
                    frac = 0.2 + 0.75 * (step / max(total, 1))
                    self.after(0, lambda m=msg, f=frac: (
                        self._set_progress(m, COLORS["warning"]),
                        self.progress_bar.set(f),
                        self.progress_detail.configure(text=f"Model {step}/{total}"),
                    ))

                # Filter SOTA_MODELS to only enabled ones
                from har.sota_benchmark import SOTA_MODELS as ALL_MODELS
                import har.sota_benchmark as sb
                original = sb.SOTA_MODELS.copy()
                sb.SOTA_MODELS = {k: v for k, v in ALL_MODELS.items() if k in enabled}

                sota_results = run_sota_benchmark(
                    frames, device=device,
                    progress_callback=on_progress,
                    cancel_check=lambda: self._cancel_flag,
                )

                sb.SOTA_MODELS = original  # Restore

                # ── Step 5: Display results ──
                self.after(0, lambda: self._show_full_results(our_result, sota_results, source_display))

                # ── Step 6: Generate report ──
                report = generate_benchmark_report(our_result, sota_results, source_display)
                self._report_text = report
                self._our_result = our_result
                self._sota_results = sota_results

                self.after(0, lambda: self.progress_bar.set(1.0))
                self.after(0, lambda: self._set_progress("Benchmark complete!", COLORS["success"]))
                self.after(0, lambda: self.export_md_btn.configure(state="normal"))
                self.after(0, lambda: self.export_pdf_btn.configure(state="normal"))

            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda msg=err_msg: self._set_progress(f"Error: {msg}", COLORS["error"]))

            finally:
                self.after(0, self._unlock_ui)

        threading.Thread(target=run, daemon=True).start()

    def _cancel_benchmark(self):
        self._cancel_flag = True
        self.cancel_btn.configure(state="disabled")
        self._set_progress("Cancelling after current model finishes...", COLORS["warning"])

    def _unlock_ui(self):
        self._benchmark_running = False
        self.start_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")

    def _set_progress(self, text, color=None):
        self.progress_label.configure(text=text,
                                       text_color=color or COLORS["text_dim"])

    # ──────────────────────────────────────────────
    # Results Display
    # ──────────────────────────────────────────────
    def _add_our_result(self, our_result):
        """Show our model result card immediately (before SOTA models finish)."""
        card = ctk.CTkFrame(self.results_scroll, fg_color=COLORS["bg"],
                            corner_radius=10, border_width=1,
                            border_color=COLORS["success"])
        card.pack(fill="x", padx=5, pady=(5, 3))

        # Header row
        hdr = ctk.CTkFrame(card, fg_color="transparent")
        hdr.pack(fill="x", padx=12, pady=(10, 2))

        ctk.CTkLabel(hdr, text="Ours — Compact R(2+1)D",
                     font=("Segoe UI", 13, "bold"),
                     text_color=COLORS["success"]).pack(side="left")
        ctk.CTkLabel(hdr, text=f"{our_result['params']}  ·  {our_result['latency_ms']:.0f} ms",
                     font=("Segoe UI", 10),
                     text_color=COLORS["text_dim"]).pack(side="right")

        # Prediction
        pred_frame = ctk.CTkFrame(card, fg_color="transparent")
        pred_frame.pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkLabel(pred_frame, text=our_result["label"],
                     font=("Segoe UI", 20, "bold"),
                     text_color=COLORS["text"]).pack(side="left")
        ctk.CTkLabel(pred_frame, text=f"{our_result['confidence']:.1%}",
                     font=("Segoe UI", 16, "bold"),
                     text_color=COLORS["success"]).pack(side="right")

        # Probability bars
        if our_result.get("all_probs"):
            sorted_probs = sorted(our_result["all_probs"].items(),
                                  key=lambda x: x[1], reverse=True)[:5]
            for cls, prob in sorted_probs:
                row = ctk.CTkFrame(card, fg_color="transparent")
                row.pack(fill="x", padx=12, pady=1)
                ctk.CTkLabel(row, text=cls, font=("Segoe UI", 9),
                             text_color=COLORS["text"], width=110,
                             anchor="w").pack(side="left")
                bar = ctk.CTkProgressBar(row, width=120, height=6,
                                         progress_color=COLORS["success"],
                                         fg_color=COLORS["progress_bg"])
                bar.pack(side="left", padx=5)
                bar.set(prob)
                ctk.CTkLabel(row, text=f"{prob:.1%}", font=("Segoe UI", 9),
                             text_color=COLORS["text_dim"], width=40).pack(side="left")

        # Bottom spacer
        ctk.CTkFrame(card, height=4, fg_color="transparent").pack()

    def _add_sota_result(self, name, data):
        """Add a single SOTA model result card."""
        is_error = data.get("status") == "error"
        border_color = COLORS["error"] if is_error else COLORS["border"]

        card = ctk.CTkFrame(self.results_scroll, fg_color=COLORS["bg"],
                            corner_radius=10, border_width=1,
                            border_color=border_color)
        card.pack(fill="x", padx=5, pady=2)

        # Header
        hdr = ctk.CTkFrame(card, fg_color="transparent")
        hdr.pack(fill="x", padx=12, pady=(8, 2))

        ctk.CTkLabel(hdr, text=name, font=("Segoe UI", 12, "bold"),
                     text_color=COLORS["text"]).pack(side="left")
        meta = f"{data['params']}  ·  {data['year']}  ·  {data.get('type', '')}"
        ctk.CTkLabel(hdr, text=meta, font=("Segoe UI", 9),
                     text_color=COLORS["text_dim"]).pack(side="right")

        if is_error:
            ctk.CTkLabel(card, text=f"Error: {data.get('error_msg', 'Unknown')}",
                         font=("Segoe UI", 10), text_color=COLORS["error"],
                         wraplength=350).pack(padx=12, pady=(0, 8), anchor="w")
            return

        # Prediction
        pred_frame = ctk.CTkFrame(card, fg_color="transparent")
        pred_frame.pack(fill="x", padx=12, pady=(0, 2))

        ctk.CTkLabel(pred_frame, text=data["label"],
                     font=("Segoe UI", 14, "bold"),
                     text_color=COLORS["text"]).pack(side="left")

        conf = data["confidence"]
        conf_color = COLORS["success"] if conf > 0.7 else (COLORS["warning"] if conf > 0.3 else COLORS["error"])
        ctk.CTkLabel(pred_frame, text=f"{conf:.1%}",
                     font=("Segoe UI", 13, "bold"),
                     text_color=conf_color).pack(side="right")

        # Latency
        ctk.CTkLabel(card, text=f"Latency: {data['latency_ms']:.0f} ms  ·  Paper accuracy: {data.get('paper_acc', 'N/A')}",
                     font=("Segoe UI", 9), text_color=COLORS["text_dim"]).pack(
            padx=12, pady=(0, 2), anchor="w")

        # Top-5 predictions
        if data.get("top5"):
            for pred in data["top5"][:3]:  # Show top-3 for compact view
                row = ctk.CTkFrame(card, fg_color="transparent")
                row.pack(fill="x", padx=12, pady=1)
                lbl = pred["label"]
                if len(lbl) > 25:
                    lbl = lbl[:23] + ".."
                ctk.CTkLabel(row, text=lbl, font=("Segoe UI", 9),
                             text_color=COLORS["text"], width=140,
                             anchor="w").pack(side="left")
                bar = ctk.CTkProgressBar(row, width=100, height=5,
                                         progress_color=COLORS["accent"],
                                         fg_color=COLORS["progress_bg"])
                bar.pack(side="left", padx=5)
                bar.set(pred["confidence"])
                ctk.CTkLabel(row, text=f"{pred['confidence']:.1%}",
                             font=("Segoe UI", 9),
                             text_color=COLORS["text_dim"], width=40).pack(side="left")

        ctk.CTkFrame(card, height=3, fg_color="transparent").pack()

    def _show_full_results(self, our_result, sota_results, source_display):
        """Display the complete benchmark results with summary table."""
        # Clear any loading indicators (keep our result card)
        children = self.results_scroll.winfo_children()
        # Keep the first child (our result card), remove the rest
        for w in children[1:]:
            w.destroy()

        # Add SOTA result cards
        for name, data in sota_results.items():
            self._add_sota_result(name, data)

        # ── Summary comparison table ──
        ctk.CTkFrame(self.results_scroll, height=1,
                     fg_color=COLORS["border"]).pack(fill="x", padx=5, pady=10)

        summary_card = ctk.CTkFrame(self.results_scroll, fg_color=COLORS["bg"],
                                     corner_radius=10, border_width=1,
                                     border_color=COLORS["accent"])
        summary_card.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(summary_card, text="Summary Comparison",
                     font=("Segoe UI", 13, "bold"),
                     text_color=COLORS["accent"]).pack(anchor="w", padx=12, pady=(10, 5))

        # Table header
        table_hdr = ctk.CTkFrame(summary_card, fg_color=COLORS["card"], corner_radius=4)
        table_hdr.pack(fill="x", padx=12, pady=2)
        for txt, w in [("Model", 95), ("Prediction", 105), ("Conf", 45),
                       ("Latency", 55), ("Params", 50), ("GFLOPs", 65)]:
            ctk.CTkLabel(table_hdr, text=txt, font=("Segoe UI", 9, "bold"),
                         text_color=COLORS["text_dim"], width=w,
                         anchor="w").pack(side="left", padx=2, pady=4)

        # Our row
        our_row = ctk.CTkFrame(summary_card, fg_color="transparent")
        our_row.pack(fill="x", padx=12, pady=1)
        for txt, w, clr in [
            ("Ours R(2+1)D", 95, COLORS["success"]),
            (our_result["label"][:13], 105, COLORS["success"]),
            (f"{our_result['confidence']:.0%}", 45, COLORS["success"]),
            (f"{our_result['latency_ms']:.0f}ms", 55, COLORS["success"]),
            (our_result["params"], 50, COLORS["success"]),
            (our_result.get("flops", "0.47G"), 65, COLORS["success"]),
        ]:
            ctk.CTkLabel(our_row, text=txt, font=("Segoe UI", 10, "bold"),
                         text_color=clr, width=w, anchor="w").pack(side="left", padx=2)

        # SOTA rows
        for name, data in sota_results.items():
            row = ctk.CTkFrame(summary_card, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=1)
            clr = COLORS["text"] if data.get("status") == "success" else COLORS["error"]
            for txt, w in [
                (name[:13], 95),
                (data["label"][:13] if data.get("status") == "success" else "Error", 105),
                (f"{data['confidence']:.0%}" if data.get("status") == "success" else "-", 45),
                (f"{data['latency_ms']:.0f}ms" if data['latency_ms'] > 0 else "-", 55),
                (data["params"], 50),
                (data.get("flops", "N/A"), 65),
            ]:
                ctk.CTkLabel(row, text=txt, font=("Segoe UI", 10),
                             text_color=clr, width=w, anchor="w").pack(side="left", padx=2)

        # Efficiency highlight
        our_params = our_result.get("params_raw", 279683)
        successful = {k: v for k, v in sota_results.items() if v.get("status") == "success"}
        if successful:
            max_params_name = max(successful, key=lambda k: float(successful[k]["params"].replace("M", "")) * 1e6)
            max_ratio = float(successful[max_params_name]["params"].replace("M", "")) * 1e6 / max(our_params, 1)

            highlight = ctk.CTkFrame(summary_card, fg_color=COLORS["card"],
                                     corner_radius=8)
            highlight.pack(fill="x", padx=12, pady=(8, 10))

            ctk.CTkLabel(highlight,
                         text=f"Our model is {max_ratio:.0f}x smaller than {max_params_name} "
                              f"({our_result['params']} vs {successful[max_params_name]['params']})",
                         font=("Segoe UI", 11, "bold"),
                         text_color=COLORS["accent"]).pack(padx=10, pady=8)

        ctk.CTkFrame(summary_card, height=4, fg_color="transparent").pack()

        # ── Historical Reference Table ──
        from har.sota_benchmark import REFERENCE_MODELS
        if REFERENCE_MODELS:
            ctk.CTkFrame(self.results_scroll, height=1,
                         fg_color=COLORS["border"]).pack(fill="x", padx=5, pady=6)

            ref_card = ctk.CTkFrame(self.results_scroll, fg_color=COLORS["bg"],
                                     corner_radius=10, border_width=1,
                                     border_color=COLORS["text_dim"])
            ref_card.pack(fill="x", padx=5, pady=5)

            ctk.CTkLabel(ref_card, text="Historical Reference (Published Accuracy)",
                         font=("Segoe UI", 12, "bold"),
                         text_color=COLORS["text_dim"]).pack(anchor="w", padx=12, pady=(10, 5))

            # Header
            ref_hdr = ctk.CTkFrame(ref_card, fg_color=COLORS["card"], corner_radius=4)
            ref_hdr.pack(fill="x", padx=12, pady=2)
            for txt, w in [("Model", 95), ("Year", 35), ("Type", 105), ("UCF-101", 50), ("Params", 45), ("GFLOPs", 55)]:
                ctk.CTkLabel(ref_hdr, text=txt, font=("Segoe UI", 8, "bold"),
                             text_color=COLORS["text_dim"], width=w,
                             anchor="w").pack(side="left", padx=2, pady=3)

            # Our row in reference context
            our_ref = ctk.CTkFrame(ref_card, fg_color="transparent")
            our_ref.pack(fill="x", padx=12, pady=1)
            for txt, w in [("Ours R(2+1)D", 95), ("2026", 35),
                           ("(2+1)D + GAP", 105),
                           (our_result.get("paper_acc", "93.83%"), 50),
                           (our_result["params"], 45),
                           (our_result.get("flops", "0.47G"), 55)]:
                ctk.CTkLabel(our_ref, text=txt, font=("Segoe UI", 9, "bold"),
                             text_color=COLORS["success"], width=w,
                             anchor="w").pack(side="left", padx=2)

            # Reference rows
            for ref_name, ref in REFERENCE_MODELS.items():
                ref_row = ctk.CTkFrame(ref_card, fg_color="transparent")
                ref_row.pack(fill="x", padx=12, pady=1)
                for txt, w in [(ref_name, 95), (str(ref["year"]), 35),
                               (ref["type"], 105), (ref["paper_acc"], 50),
                               (ref["params"], 45), (ref.get("flops", "N/A"), 55)]:
                    ctk.CTkLabel(ref_row, text=txt, font=("Segoe UI", 9),
                                 text_color=COLORS["text_dim"], width=w,
                                 anchor="w").pack(side="left", padx=2)

            ctk.CTkFrame(ref_card, height=4, fg_color="transparent").pack()

    # ──────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────
    def _export_report(self, fmt):
        if fmt == "md":
            path = filedialog.asksaveasfilename(
                defaultextension=".md",
                filetypes=[("Markdown", "*.md"), ("All", "*.*")],
                initialfile="sota_benchmark_report.md")
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self._report_text)
                self._set_progress(f"Report exported: {path}", COLORS["success"])

        elif fmt == "pdf":
            path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF", "*.pdf"), ("All", "*.*")],
                initialfile="sota_benchmark_report.pdf")
            if path:
                self._export_pdf(path)

    def _export_pdf(self, output_path):
        """Generate a professional PDF benchmark report."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_JUSTIFY
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            )

            doc = SimpleDocTemplate(output_path, pagesize=letter,
                                    leftMargin=54, rightMargin=54,
                                    topMargin=60, bottomMargin=60)

            # Styles
            primary = colors.HexColor("#1e293b")
            accent = colors.HexColor("#3b82f6")
            text_c = colors.HexColor("#334155")
            bg_light = colors.HexColor("#f8fafc")

            title_s = ParagraphStyle('T', fontName='Helvetica-Bold',
                                     fontSize=18, leading=24, textColor=primary)
            h1_s = ParagraphStyle('H1', fontName='Helvetica-Bold',
                                  fontSize=13, leading=17, textColor=primary,
                                  spaceBefore=14, spaceAfter=6)
            body_s = ParagraphStyle('B', fontName='Helvetica',
                                    fontSize=9.5, leading=13.5, textColor=text_c,
                                    spaceAfter=7, alignment=TA_JUSTIFY)
            hdr_s = ParagraphStyle('TH', fontName='Helvetica-Bold',
                                   fontSize=8, textColor=colors.white, leading=11)
            cell_s = ParagraphStyle('TD', fontName='Helvetica',
                                   fontSize=8, textColor=text_c, leading=11)
            bold_s = ParagraphStyle('TDB', fontName='Helvetica-Bold',
                                   fontSize=8, textColor=primary, leading=11)

            story = []
            story.append(Paragraph("SOTA Benchmark Report", title_s))
            story.append(Spacer(1, 8))

            import torch
            dev = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
            story.append(Paragraph(
                f"<b>Date:</b> {time.strftime('%Y-%m-%d %H:%M')}<br/>"
                f"<b>Device:</b> {dev}", body_s))
            story.append(Spacer(1, 12))

            # Comparison Table
            story.append(Paragraph("Comparison Table", h1_s))

            header = [
                Paragraph("<b>Model</b>", hdr_s),
                Paragraph("<b>Prediction</b>", hdr_s),
                Paragraph("<b>Conf</b>", hdr_s),
                Paragraph("<b>Latency</b>", hdr_s),
                Paragraph("<b>Params</b>", hdr_s),
                Paragraph("<b>Paper Acc</b>", hdr_s),
                Paragraph("<b>GFLOPs</b>", hdr_s),
            ]

            data = [header]
            # Our row
            our = self._our_result
            data.append([
                Paragraph(f"<b>Ours R(2+1)D</b>", bold_s),
                Paragraph(f"<b>{our['label']}</b>", bold_s),
                Paragraph(f"<b>{our['confidence']:.0%}</b>", bold_s),
                Paragraph(f"<b>{our['latency_ms']:.0f}ms</b>", bold_s),
                Paragraph(f"<b>{our['params']}</b>", bold_s),
                Paragraph(f"<b>{our.get('paper_acc', 'N/A')}</b>", bold_s),
                Paragraph(f"<b>{our.get('flops', '0.47G')}</b>", bold_s),
            ])

            for name, d in self._sota_results.items():
                if d.get("status") == "success":
                    data.append([
                        Paragraph(name, cell_s),
                        Paragraph(d["label"][:20], cell_s),
                        Paragraph(f"{d['confidence']:.0%}", cell_s),
                        Paragraph(f"{d['latency_ms']:.0f}ms", cell_s),
                        Paragraph(d["params"], cell_s),
                        Paragraph(d.get("paper_acc", "N/A"), cell_s),
                        Paragraph(d.get("flops", "N/A"), cell_s),
                    ])
                else:
                    data.append([
                        Paragraph(name, cell_s),
                        Paragraph("Error", cell_s),
                        Paragraph("-", cell_s),
                        Paragraph("-", cell_s),
                        Paragraph(d["params"], cell_s),
                        Paragraph(d.get("paper_acc", "N/A"), cell_s),
                        Paragraph(d.get("flops", "N/A"), cell_s),
                    ])

            cw = [80, 90, 35, 50, 45, 55, 50]
            t = Table(data, colWidths=cw)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), primary),
                ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#dbeafe")),
                ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ('ROWBACKGROUNDS', (0, 2), (-1, -1), [colors.white, bg_light]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(t)
            story.append(Spacer(1, 14))

            story.append(Paragraph(
                f"Generated by HAR Control Center SOTA Benchmark Engine", body_s))

            doc.build(story)
            self._set_progress(f"PDF exported: {output_path}", COLORS["success"])

        except ImportError:
            self._set_progress("Install reportlab for PDF export: pip install reportlab",
                               COLORS["error"])
        except Exception as e:
            self._set_progress(f"PDF export error: {e}", COLORS["error"])
