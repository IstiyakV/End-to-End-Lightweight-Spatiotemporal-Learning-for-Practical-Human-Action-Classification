"""Model Tester — upload video, predict, Grad-CAM, YouTube stream detection."""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import threading
import sys

from gui.theme import COLORS


class CTkConsoleLabel(ctk.CTkFrame):
    def __init__(self, master, font=("Segoe UI", 11), text_color=None, **kwargs):
        super().__init__(master, fg_color=COLORS["bg"], corner_radius=8, border_width=1, border_color=COLORS["border"], **kwargs)
        self.textbox = ctk.CTkTextbox(self, font=("Consolas", 10), height=55, fg_color="transparent", text_color=text_color, wrap="word")
        self.textbox.pack(fill="both", expand=True, padx=5, pady=5)
        self.textbox.configure(state="disabled")
        
    def configure(self, text=None, text_color=None, **kwargs):
        if text is not None:
            self.textbox.configure(state="normal")
            self.textbox.delete("1.0", "end")
            self.textbox.insert("1.0", str(text))
            self.textbox.configure(state="disabled")
            self.textbox.see("end")
        if text_color is not None:
            self.textbox.configure(text_color=text_color)
        super().configure(**kwargs)


class ModelTesterFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=COLORS["bg"], corner_radius=0)
        self.app = app
        self.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(self, text="Model Tester", font=("Segoe UI", 24, "bold"),
                     text_color=COLORS["text"]).grid(row=0, column=0, columnspan=2,
                                                      sticky="w", padx=30, pady=(25, 15))

        # Left: controls
        left = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=12,
                            border_width=1, border_color=COLORS["border"])
        left.grid(row=1, column=0, sticky="nsew", padx=(30, 10), pady=(0, 20))

        # Model Selector (Universally used at the top)
        ctk.CTkLabel(left, text="Select Model Checkpoint", font=("Segoe UI", 13, "bold"),
                     text_color=COLORS["accent"]).pack(anchor="w", padx=20, pady=(15, 5))

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from har import config as hcfg
        self.hcfg = hcfg

        checkpoints = [p.name for p in hcfg.CHECKPOINT_DIR.glob("*_best.pth")]
        self.model_var = ctk.StringVar(value=checkpoints[0] if checkpoints else "No models")
        
        # Model Selection Frame (Horizontal layout to include Load/Import button)
        model_select_frame = ctk.CTkFrame(left, fg_color="transparent")
        model_select_frame.pack(fill="x", padx=20, pady=5)
        
        self.model_menu = ctk.CTkOptionMenu(model_select_frame, variable=self.model_var, values=checkpoints or ["No models"],
                          font=("Segoe UI", 11), fg_color=COLORS["input_bg"],
                          button_color=COLORS["accent"])
        self.model_menu.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.import_model_btn = ctk.CTkButton(model_select_frame, text="📥 Load File", width=85, height=28,
                          font=("Segoe UI", 11, "bold"), fg_color=COLORS["accent"],
                          text_color=("#000", "#000"), corner_radius=6,
                          command=self._import_model)
        self.import_model_btn.pack(side="right")

        # Segmented Control / Tabview inside left panel
        self.tabview = ctk.CTkTabview(left, fg_color="transparent", height=320, corner_radius=10,
                                       segmented_button_selected_color=COLORS["accent"],
                                       segmented_button_selected_hover_color=COLORS["accent"])
        self.tabview.pack(fill="both", expand=True, padx=15, pady=(10, 5))
        
        tab_local = self.tabview.add("Local Video")
        tab_yt = self.tabview.add("YouTube Stream")

        # Tab 1: Local Video controls
        ctk.CTkLabel(tab_local, text="Video File Path", font=("Segoe UI", 11, "bold"),
                     text_color=COLORS["accent"], anchor="w").pack(fill="x", padx=10, pady=(5, 2))
        
        self.video_path_var = ctk.StringVar(value="No file selected")
        ctk.CTkLabel(tab_local, textvariable=self.video_path_var, font=("Segoe UI", 10),
                     text_color=COLORS["text_dim"], wraplength=260, anchor="w", justify="left").pack(fill="x", padx=10, pady=2)

        ctk.CTkButton(tab_local, text="Browse Video File", font=("Segoe UI", 11),
                       fg_color=COLORS["input_bg"], hover_color=COLORS["border"], border_width=1, border_color=COLORS["border"],
                       text_color=COLORS["text"], corner_radius=8, height=30,
                       command=self._browse_video).pack(padx=10, pady=5, fill="x")

        self.gradcam_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(tab_local, text="Generate Grad-CAM", variable=self.gradcam_var,
                       progress_color=COLORS["accent"], font=("Segoe UI", 11)).pack(
            anchor="w", padx=10, pady=5)

        ctk.CTkButton(tab_local, text="Run Local Prediction", font=("Segoe UI", 12, "bold"),
                      fg_color=COLORS["success"], hover_color=COLORS["success"],
                      corner_radius=8, height=36, command=self._predict).pack(
            padx=10, pady=(10, 5), fill="x")

        # Tab 2: YouTube URL controls
        ctk.CTkLabel(tab_yt, text="YouTube Video Stream URL", font=("Segoe UI", 11, "bold"),
                     text_color=COLORS["accent"], anchor="w").pack(fill="x", padx=10, pady=(5, 2))

        self.yt_url_var = ctk.StringVar(value="https://www.youtube.com/watch?v=Tbv1a5vYI24")
        self.yt_entry = ctk.CTkEntry(tab_yt, textvariable=self.yt_url_var, font=("Segoe UI", 10),
                                     fg_color=COLORS["input_bg"], border_color=COLORS["border"],
                                     height=32, placeholder_text="Paste YouTube link here...")
        self.yt_entry.pack(fill="x", padx=10, pady=5)

        # YouTube suggestions panel
        ctk.CTkLabel(tab_yt, text="💡 Test Suggestions (Click to load & copy):", 
                     font=("Segoe UI", 11, "bold"), text_color=COLORS["accent"], anchor="w").pack(fill="x", padx=10, pady=(10, 2))
        
        suggest_frame = ctk.CTkFrame(tab_yt, fg_color="transparent")
        suggest_frame.pack(fill="x", padx=10, pady=5)
        suggest_frame.grid_columnconfigure((0, 1), weight=1)
        
        suggestions = [
            ("Clip 1 (Action)", "https://www.youtube.com/watch?v=wOEKdWrtz6U"),
            ("Clip 2 (Sports)", "https://www.youtube.com/watch?v=wIYD42DV3Ro"),
            ("Clip 3 (Dance)", "https://www.youtube.com/watch?v=msXtQTh81jA"),
            ("Clip 4 (Gym)", "https://www.youtube.com/watch?v=EnBQcffEKLc"),
            ("Clip 5 (Run)", "https://www.youtube.com/watch?v=zVqvd6mhat8"),
            ("Clip 6 (Fit)", "https://www.youtube.com/watch?v=wEVAlMTeyWc")
        ]
        
        for idx, (label, url) in enumerate(suggestions):
            row = idx // 2
            col = idx % 2
            btn = ctk.CTkButton(suggest_frame, text=label, font=("Segoe UI", 10),
                                fg_color=COLORS["input_bg"], hover_color=COLORS["border"],
                                border_width=1, border_color=COLORS["border"],
                                text_color=COLORS["text"], corner_radius=6, height=26,
                                command=lambda u=url: self._select_suggested_yt(u))
            btn.grid(row=row, column=col, padx=4, pady=3, sticky="ew")

        ctk.CTkLabel(tab_yt, text="Note: Extracting stream metadata requires a stable internet connection.", 
                     font=("Segoe UI", 9), text_color=COLORS["text_dim"], wraplength=260, justify="left").pack(fill="x", padx=10, pady=5)

        self.yt_btn = ctk.CTkButton(tab_yt, text="Stream & Detect Live", font=("Segoe UI", 12, "bold"),
                      fg_color=COLORS["accent"], hover_color=COLORS["accent"],
                      corner_radius=8, height=36, command=self._start_youtube)
        self.yt_btn.pack(padx=10, pady=(15, 5), fill="x")

        # Universally applicable Hardware Webcam test (at the very bottom of left panel)
        ctk.CTkFrame(left, height=1, fg_color=COLORS["border"]).pack(fill="x", padx=20, pady=5)
        self.webcam_btn = ctk.CTkButton(left, text="Real Time Test (Hardware Webcam)", font=("Segoe UI", 12, "bold"),
                      fg_color="#e67e22", hover_color="#d35400",
                      corner_radius=10, height=40, command=self._start_realtime)
        self.webcam_btn.pack(padx=20, pady=(5, 15), fill="x")

        # Real-time state flags
        self.realtime_running = False
        self.realtime_stop_event = None

        # PIL GIF Animator state variables
        self.gif_timer_id = None
        self.gif_frames = []
        self.gif_frame_idx = 0

        # Right: results
        right = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=12,
                             border_width=1, border_color=COLORS["border"])
        right.grid(row=1, column=1, sticky="nsew", padx=(10, 30), pady=(0, 20))
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure((0, 1), weight=1)

        # Header across both columns
        ctk.CTkLabel(right, text="Prediction Analysis & Visualisation", font=("Segoe UI", 14, "bold"),
                     text_color=COLORS["accent"]).grid(row=0, column=0, columnspan=2, sticky="w", padx=20, pady=(15, 5))

        # Column 0: Text Metrics & Probabilities
        col_left = ctk.CTkFrame(right, fg_color="transparent")
        col_left.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=10)

        self.result_class = ctk.CTkLabel(col_left, text="-", font=("Segoe UI", 28, "bold"),
                                          text_color=COLORS["text"], anchor="w")
        self.result_class.pack(fill="x", pady=2)

        self.result_conf = ctk.CTkLabel(col_left, text="Confidence: -", font=("Segoe UI", 13),
                                         text_color=COLORS["success"], anchor="w")
        self.result_conf.pack(fill="x", pady=2)

        self.result_latency = ctk.CTkLabel(col_left, text="Latency: -", font=("Segoe UI", 11),
                                            text_color=COLORS["text_dim"], anchor="w")
        self.result_latency.pack(fill="x", pady=2)

        ctk.CTkLabel(col_left, text="Class Probabilities (Top-5)", font=("Segoe UI", 11, "bold"),
                     text_color=COLORS["text"], anchor="w").pack(fill="x", pady=(15, 2))

        self.probs_frame = ctk.CTkFrame(col_left, fg_color="transparent")
        self.probs_frame.pack(fill="both", expand=True, pady=2)

        # Stream Session Analytics (Chronological Action Timeline & Cumulative leaderboard)
        ctk.CTkLabel(col_left, text="📊 Stream Session Analytics", font=("Segoe UI", 11, "bold"),
                     text_color=COLORS["accent"], anchor="w").pack(fill="x", pady=(15, 2))

        # Helper method to resolve appearance-dependent colors dynamically in Tkinter components
        def _resolve(color_tuple):
            idx = 1 if ctk.get_appearance_mode().lower() == "dark" else 0
            return color_tuple[idx] if isinstance(color_tuple, tuple) else color_tuple

        # 1. Temporal Action Timeline
        self.timeline_title = ctk.CTkLabel(col_left, text="Chronological Activity Timeline", font=("Segoe UI", 9, "bold"),
                                           text_color=COLORS["text_dim"], anchor="w")
        self.timeline_title.pack(fill="x", pady=(2, 1))

        self.timeline_canvas = tk.Canvas(
            col_left, height=16, highlightthickness=1,
            highlightbackground=_resolve(COLORS["border"]),
            bg=_resolve(COLORS["bg"])
        )
        self.timeline_canvas.pack(fill="x", pady=2)

        # 2. Cumulative Share Leaderboard Frame
        self.leaderboard_title = ctk.CTkLabel(col_left, text="Session Primary Action Share (Cumulative)", font=("Segoe UI", 9, "bold"),
                                              text_color=COLORS["text_dim"], anchor="w")
        self.leaderboard_title.pack(fill="x", pady=(5, 1))

        self.share_frame = ctk.CTkFrame(col_left, fg_color="transparent")
        self.share_frame.pack(fill="x", pady=2)

        # Column 1: Grad-CAM Preview
        self.col_right = ctk.CTkFrame(right, fg_color=COLORS["bg"], corner_radius=10, border_width=1, border_color=COLORS["border"])
        self.col_right.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=10)
        self.col_right.grid_rowconfigure(0, weight=1)
        self.col_right.grid_columnconfigure(0, weight=1)

        self.gif_label = ctk.CTkLabel(
            self.col_right, 
            text="Enable 'Generate Grad-CAM'\nto view live spatiotemporal\nattention preview here.",
            font=("Segoe UI", 10),
            text_color=COLORS["text_dim"]
        )
        self.gif_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Status Bar across bottom
        self.status_lbl = CTkConsoleLabel(right, text_color=COLORS["text_dim"])
        self.status_lbl.grid(row=2, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 15))

    def _stop_realtime(self):
        if self.realtime_stop_event:
            self.realtime_stop_event.set()
        self.realtime_running = False
        self.realtime_stop_event = None

    def _start_realtime(self):
        self._stop_gif()
        if self.realtime_running:
            self._stop_realtime()
            self.status_lbl.configure(text="Webcam inference stopped.", text_color=COLORS["text_dim"])
            self.webcam_btn.configure(text="Real Time Test (Hardware Webcam)", fg_color="#e67e22")
            self.yt_btn.configure(state="normal")
            return

        model_name = self.model_var.get()
        if "No" in model_name:
            self.status_lbl.configure(text="Select model first", text_color=COLORS["error"])
            return

        self.status_lbl.configure(text="Starting Real Time Webcam Inference...", text_color=COLORS["warning"])
        self.realtime_running = True
        self.session_tally = {}
        self.session_timeline = []
        self.class_colors = {}
        if hasattr(self, "timeline_canvas"):
            self.timeline_canvas.delete("all")
        import threading
        self.realtime_stop_event = threading.Event()
        self.webcam_btn.configure(text="🛑 Stop Real-Time Test", fg_color=COLORS["error"])
        self.yt_btn.configure(state="disabled")

        def frame_cb(frame, pred_class, pred_conf, fps=None, latency_ms=None, all_probs=None):
            if not self.realtime_running:
                return
            import cv2
            import PIL.Image
            from customtkinter import CTkImage
            
            # Convert BGR frame to RGB and fit cleanly in our col_right preview display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = PIL.Image.fromarray(rgb)
            
            # Get container dimensions dynamically to fill the panel!
            max_w = self.col_right.winfo_width() - 20
            max_h = self.col_right.winfo_height() - 20
            
            # Default fallbacks if window hasn't drawn fully yet
            if max_w < 100: max_w = 480
            if max_h < 100: max_h = 360
            
            orig_w, orig_h = img_pil.size
            aspect = orig_w / orig_h
            
            if aspect > (max_w / max_h):  # Landscape/wide relative to bounding box
                new_w = max_w
                new_h = int(max_w / aspect)
            else:  # Portrait/tall relative to bounding box
                new_h = max_h
                new_w = int(max_h * aspect)
                
            img_resized = img_pil.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
            ctk_img = CTkImage(light_image=img_resized, dark_image=img_resized, size=(new_w, new_h))
            
            fps_val = f"{fps:.0f}" if fps is not None else "-"
            lat_val = f"{latency_ms:.0f} ms" if (latency_ms is not None and latency_ms > 0) else "-"
            
            self.after(0, lambda: (
                self.gif_label.configure(image=ctk_img, text=""),
                self.result_class.configure(text=pred_class),
                self.result_conf.configure(text=f"Confidence: {pred_conf:.1%}"),
                self.result_latency.configure(text=f"Latency: {lat_val}"),
                self.status_lbl.configure(text=f"Streaming live... Processing: {fps_val} FPS", text_color=COLORS["success"]),
                self._update_probs_ui(all_probs),
                self._process_frame_telemetry(pred_class, all_probs)
            ))

        def run():
            try:
                from har.predict import load_model, predict_realtime
                
                ckpt_path = str(self.hcfg.CHECKPOINT_DIR / model_name)
                model, class_names = load_model(ckpt_path, "cuda" if __import__("torch").cuda.is_available() else "cpu")
                
                self.after(0, lambda: self.status_lbl.configure(
                    text="Webcam active. Rendering live stream in right panel...", text_color=COLORS["success"]))

                predict_realtime(model, class_names, 
                                 device="cuda" if __import__("torch").cuda.is_available() else "cpu",
                                 stop_event=self.realtime_stop_event,
                                 frame_callback=frame_cb)

                self.after(0, lambda: (
                    self.status_lbl.configure(text="Webcam inference closed.", text_color=COLORS["text_dim"]),
                    self.webcam_btn.configure(text="Real Time Test (Hardware Webcam)", fg_color="#e67e22"),
                    self.yt_btn.configure(state="normal")
                ))

            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda msg=err_msg: (
                    self.status_lbl.configure(text=f"Error: {msg}", text_color=COLORS["error"]),
                    self.webcam_btn.configure(text="Real Time Test (Hardware Webcam)", fg_color="#e67e22"),
                    self.yt_btn.configure(state="normal")
                ))

        threading.Thread(target=run, daemon=True).start()

    def _select_suggested_yt(self, url):
        self.yt_url_var.set(url)
        try:
            self.clipboard_clear()
            self.clipboard_append(url)
            self.status_lbl.configure(
                text="✓ Loaded YouTube URL & copied to system clipboard!",
                text_color=COLORS["success"]
            )
        except Exception as e:
            self.status_lbl.configure(
                text=f"Loaded YouTube URL (Clipboard copy failed: {e})",
                text_color=COLORS["warning"]
            )

    def _start_youtube(self):
        self._stop_gif()
        if self.realtime_running:
            self._stop_realtime()
            self.status_lbl.configure(text="YouTube stream stopped.", text_color=COLORS["text_dim"])
            self.yt_btn.configure(text="Stream & Detect Live", fg_color=COLORS["accent"])
            self.webcam_btn.configure(state="normal")
            return

        model_name = self.model_var.get()
        yt_url = self.yt_url_var.get().strip()

        if "No" in model_name or not yt_url:
            self.status_lbl.configure(text="Select model and paste a valid YouTube URL first", text_color=COLORS["error"])
            return

        self.status_lbl.configure(text="Extracting YouTube stream metadata...", text_color=COLORS["warning"])
        self.realtime_running = True
        self.session_tally = {}
        self.session_timeline = []
        self.class_colors = {}
        if hasattr(self, "timeline_canvas"):
            self.timeline_canvas.delete("all")
        import threading
        self.realtime_stop_event = threading.Event()
        self.yt_btn.configure(text="🛑 Stop Stream", fg_color=COLORS["error"])
        self.webcam_btn.configure(state="disabled")

        def frame_cb(frame, pred_class, pred_conf, fps=None, latency_ms=None, all_probs=None):
            if not self.realtime_running:
                return
            import cv2
            import PIL.Image
            from customtkinter import CTkImage
            
            # Convert BGR frame to RGB and fit cleanly in our col_right preview display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = PIL.Image.fromarray(rgb)
            
            # Get container dimensions dynamically to fill the panel!
            max_w = self.col_right.winfo_width() - 20
            max_h = self.col_right.winfo_height() - 20
            
            # Default fallbacks if window hasn't drawn fully yet
            if max_w < 100: max_w = 480
            if max_h < 100: max_h = 360
            
            orig_w, orig_h = img_pil.size
            aspect = orig_w / orig_h
            
            if aspect > (max_w / max_h):  # Landscape/wide relative to bounding box
                new_w = max_w
                new_h = int(max_w / aspect)
            else:  # Portrait/tall relative to bounding box
                new_h = max_h
                new_w = int(max_h * aspect)
                
            img_resized = img_pil.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
            ctk_img = CTkImage(light_image=img_resized, dark_image=img_resized, size=(new_w, new_h))
            
            fps_val = f"{fps:.0f}" if fps is not None else "-"
            lat_val = f"{latency_ms:.0f} ms" if (latency_ms is not None and latency_ms > 0) else "-"
            
            self.after(0, lambda: (
                self.gif_label.configure(image=ctk_img, text=""),
                self.result_class.configure(text=pred_class),
                self.result_conf.configure(text=f"Confidence: {pred_conf:.1%}"),
                self.result_latency.configure(text=f"Latency: {lat_val}"),
                self.status_lbl.configure(text=f"Streaming live... Processing: {fps_val} FPS", text_color=COLORS["success"]),
                self._update_probs_ui(all_probs),
                self._process_frame_telemetry(pred_class, all_probs)
            ))

        def run():
            try:
                from har.predict import load_model, predict_realtime
                
                ckpt_path = str(self.hcfg.CHECKPOINT_DIR / model_name)
                model, class_names = load_model(ckpt_path, "cuda" if __import__("torch").cuda.is_available() else "cpu")
                
                self.after(0, lambda: self.status_lbl.configure(
                    text="Streaming YouTube Video. Rendering live in right panel...", text_color=COLORS["success"]))

                predict_realtime(model, class_names, 
                                 device="cuda" if __import__("torch").cuda.is_available() else "cpu",
                                 video_source=yt_url,
                                 stop_event=self.realtime_stop_event,
                                 frame_callback=frame_cb)

                self.after(0, lambda: (
                    self.status_lbl.configure(text="YouTube stream inference closed.", text_color=COLORS["text_dim"]),
                    self.yt_btn.configure(text="Stream & Detect Live", fg_color=COLORS["accent"]),
                    self.webcam_btn.configure(state="normal")
                ))

            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda msg=err_msg: (
                    self.status_lbl.configure(text=f"Error: {msg}", text_color=COLORS["error"]),
                    self.yt_btn.configure(text="Stream & Detect Live", fg_color=COLORS["accent"]),
                    self.webcam_btn.configure(state="normal")
                ))

        threading.Thread(target=run, daemon=True).start()

    def _browse_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])
        if path:
            self.video_path_var.set(path)

    def _predict(self):
        self._stop_gif()
        self.gif_label.configure(image="", text="Computing Prediction...", font=("Segoe UI", 11))
        video = self.video_path_var.get()
        model_name = self.model_var.get()

        if "No" in video or "No" in model_name:
            self.status_lbl.configure(text="Select model and video first", text_color=COLORS["error"])
            return

        self.status_lbl.configure(text="Running inference...", text_color=COLORS["warning"])

        def run():
            try:
                import time
                from har.predict import load_model, predict_video

                ckpt_path = str(self.hcfg.CHECKPOINT_DIR / model_name)
                model, class_names = load_model(ckpt_path, "cuda" if __import__("torch").cuda.is_available() else "cpu")

                t0 = time.perf_counter()
                result = predict_video(model, video, class_names,
                                       "cuda" if __import__("torch").cuda.is_available() else "cpu")
                latency = (time.perf_counter() - t0) * 1000

                self.after(0, lambda: self._show_result(result, latency))

                if self.gradcam_var.get():
                    from har.gradcam import generate_gradcam_overlay
                    gif_path = self.hcfg.RESULTS_DIR / "gradcam_preview.gif"
                    
                    self.after(0, lambda: self.gif_label.configure(text="Generating Grad-CAM Heatmap..."))
                    generate_gradcam_overlay(model, video, gif_path,
                                             device="cuda" if __import__("torch").cuda.is_available() else "cpu")
                    
                    # Schedule visual animation preview inside main thread
                    self.after(0, lambda: self._animate_gif(gif_path))
                    self.after(0, lambda: self.status_lbl.configure(
                        text=f"Grad-CAM saved: {gif_path}", text_color=COLORS["success"]))

            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda msg=err_msg: self.status_lbl.configure(
                    text=f"Error: {msg}", text_color=COLORS["error"]))
                self.after(0, lambda msg=err_msg: self.gif_label.configure(text=f"Grad-CAM error: {msg}", text_color=COLORS["error"]))

        threading.Thread(target=run, daemon=True).start()

    def _show_result(self, result, latency_ms):
        self.result_class.configure(text=result["class"])
        self.result_conf.configure(text=f"Confidence: {result['confidence']:.1%}")
        self.result_latency.configure(text=f"Latency: {latency_ms:.0f} ms")
        self._update_probs_ui(result.get("all_probs", {}))
        self.status_lbl.configure(text="Prediction complete", text_color=COLORS["success"])

    def _update_probs_ui(self, all_probs):
        if not all_probs:
            return
            
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Check if we need to initialize the reusable rows
        if not hasattr(self, "prob_widgets") or len(self.prob_widgets) != 5:
            # Clear old children just in case
            for w in self.probs_frame.winfo_children():
                w.destroy()
            self.prob_widgets = []
            
            for i in range(5):
                row = ctk.CTkFrame(self.probs_frame, fg_color="transparent")
                row.pack(fill="x", pady=2)
                
                lbl = ctk.CTkLabel(row, text="-", font=("Segoe UI", 10), text_color=COLORS["text"],
                                   width=110, anchor="w")
                lbl.pack(side="left")
                
                bar = ctk.CTkProgressBar(row, width=120, height=8,
                                         progress_color=COLORS["accent"], fg_color=COLORS["progress_bg"])
                bar.pack(side="left", padx=5)
                bar.set(0.0)
                
                pct = ctk.CTkLabel(row, text="0.0%", font=("Segoe UI", 10),
                                   text_color=COLORS["text_dim"], width=40)
                pct.pack(side="left")
                
                self.prob_widgets.append((lbl, bar, pct))
                
        # Update text/progress for the top-5 classes
        for i, (cls, prob) in enumerate(sorted_probs):
            if i < len(self.prob_widgets):
                lbl, bar, pct = self.prob_widgets[i]
                lbl.configure(text=cls)
                bar.set(prob)
                pct.configure(text=f"{prob:.1%}")

    # ─────────────────────── GIF ANIMATOR METHODS ───────────────────
    def _animate_gif(self, gif_path):
        self._stop_gif()
        self.gif_frames = []
        self.gif_frame_idx = 0
        
        try:
            from PIL import Image
            from customtkinter import CTkImage
            im = Image.open(gif_path)
            
            # Load and resize each frame of the spatiotemporal overlay GIF
            try:
                while True:
                    frame = im.copy().convert("RGBA")
                    # Fit cleanly within the col_right display frame (200x200)
                    resized = frame.resize((200, 200), Image.Resampling.LANCZOS)
                    # Convert to CTkImage instead of ImageTk.PhotoImage to support clean HighDPI scaling
                    self.gif_frames.append(CTkImage(light_image=resized, dark_image=resized, size=(200, 200)))
                    im.seek(im.tell() + 1)
            except EOFError:
                pass  # Successfully read all frames
                
            if self.gif_frames:
                self.gif_label.configure(text="")
                self._play_gif()
        except Exception as e:
            self.gif_label.configure(text=f"Failed to play preview: {e}", text_color=COLORS["warning"])

    def _play_gif(self):
        if not self.gif_frames:
            return
        # Display current frame (CTkImage fits naturally here)
        self.gif_label.configure(image=self.gif_frames[self.gif_frame_idx])
        # Increment index
        self.gif_frame_idx = (self.gif_frame_idx + 1) % len(self.gif_frames)
        # Schedule next frame
        self.gif_timer_id = self.after(100, self._play_gif)

    def _stop_gif(self):
        if self.gif_timer_id:
            self.after_cancel(self.gif_timer_id)
            self.gif_timer_id = None
        self.gif_frames = []
        self.gif_frame_idx = 0
        self.gif_label.configure(image="", text="Enable 'Generate Grad-CAM'\nto view live spatiotemporal\nattention preview here.")

    def _import_model(self):
        """Open a file dialog to import a model checkpoint from anywhere on the PC."""
        self._stop_gif()
        
        # Open file browser for .pth files
        selected_path = filedialog.askopenfilename(
            title="Select External PyTorch Model Checkpoint",
            filetypes=[("PyTorch Checkpoint", "*.pth"), ("All Files", "*.*")]
        )
        if not selected_path:
            return
            
        selected_path = Path(selected_path)
        self.status_lbl.configure(text="Validating external checkpoint...", text_color=COLORS["warning"])
        self.update_idletasks()
        
        # Run validation in background/try-except to avoid GUI crashes
        try:
            import torch
            import shutil
            
            # Load with weights_only=False to read the dict fully
            ckpt = torch.load(str(selected_path), map_location="cpu", weights_only=False)
            
            # Validate essential components for the predictor
            if not isinstance(ckpt, dict):
                raise ValueError("Checkpoint is not a dictionary state dictionary.")
                
            # If direct state_dict (missing 'model_state_dict'), wrap it robustly!
            if "model_state_dict" not in ckpt:
                is_direct_state_dict = any(".weight" in k or ".bias" in k for k in ckpt.keys())
                if is_direct_state_dict:
                    raw_state_dict = ckpt
                    # Determine number of classes from last linear layer weight shape
                    num_classes = None
                    for key in ["fc.weight", "classifier.weight", "linear.weight", "classifier.1.weight"]:
                        if key in raw_state_dict:
                            num_classes = raw_state_dict[key].shape[0]
                            break
                    if num_classes is None:
                        # Fallback: find any 2D weight tensor in the last layers
                        for key in reversed(list(raw_state_dict.keys())):
                            if key.endswith(".weight") and len(raw_state_dict[key].shape) == 2:
                                num_classes = raw_state_dict[key].shape[0]
                                break
                    if num_classes is None:
                        num_classes = 101 # default fallback
                    
                    ckpt = {
                        "model_state_dict": raw_state_dict,
                        "config": {"num_classes": num_classes}
                    }
                else:
                    raise ValueError("Missing 'model_state_dict' inside the checkpoint.")
            
            # Ensure config and num_classes exist
            if "config" not in ckpt:
                ckpt["config"] = {}
            if "num_classes" not in ckpt["config"]:
                # Try to resolve classes from model state dict
                num_classes = None
                raw_state_dict = ckpt["model_state_dict"]
                for key in ["fc.weight", "classifier.weight", "linear.weight", "classifier.1.weight"]:
                    if key in raw_state_dict:
                        num_classes = raw_state_dict[key].shape[0]
                        break
                if num_classes is None:
                    num_classes = 101
                ckpt["config"]["num_classes"] = num_classes
                
            num_classes = ckpt["config"]["num_classes"]
            
            # Ask if they have a custom class mapping JSON
            import json
            import tkinter.messagebox as messagebox
            use_json = messagebox.askyesno(
                "Class Mapping JSON",
                "Do you have a JSON file mapping class numbers to real class names?\n\nIf you select Yes, you can load your JSON mapping (e.g. key-value pairs where key is class number). Otherwise, clean generic class numbers will be used."
            )
            
            class_mapping = None
            if use_json:
                json_path = filedialog.askopenfilename(
                    title="Select Class Mapping JSON File",
                    filetypes=[("JSON Mapping File", "*.json"), ("All Files", "*.*")]
                )
                if json_path:
                    try:
                        with open(json_path, "r") as fj:
                            raw_mapping = json.load(fj)
                        # Normalize keys: convert strings to integers
                        class_mapping = {}
                        for k, v in raw_mapping.items():
                            try:
                                class_mapping[int(k)] = str(v)
                            except ValueError:
                                pass
                    except Exception as ex:
                        messagebox.showerror("JSON Parsing Error", f"Failed to parse class mapping JSON: {ex}")
            
            if class_mapping:
                ckpt["class_names"] = []
                # Handle 0-based or 1-based indexing in user JSON
                is_one_based = (0 not in class_mapping) and (1 in class_mapping)
                for i in range(num_classes):
                    key = (i + 1) if is_one_based else i
                    ckpt["class_names"].append(class_mapping.get(key, f"Class {i}"))
            else:
                # Reconstruct class names if missing via local datasets
                discovered = []
                for possible_path in [self.hcfg.UCF101_DIR, self.hcfg.DATA_DIR]:
                    if possible_path and possible_path.exists():
                        try:
                            from har.dataset import discover_classes
                            classes, _ = discover_classes(possible_path)
                            if len(classes) == num_classes:
                                discovered = classes
                                break
                        except:
                            pass
                if discovered:
                    ckpt["class_names"] = discovered
                else:
                    # Clean fallback: show pure class numbers!
                    ckpt["class_names"] = [f"Class {i}" for i in range(num_classes)]
            
            # Destination directory
            self.hcfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            
            # Format filename to end with _best.pth so it matches the GUI scanner's glob
            base_name = selected_path.stem
            if base_name.endswith("_best"):
                dest_name = f"{base_name}.pth"
            elif base_name.endswith("_latest"):
                dest_name = f"{base_name[:-7]}_best.pth"
            else:
                dest_name = f"{base_name}_best.pth"
                
            dest_path = self.hcfg.CHECKPOINT_DIR / dest_name
            
            # Save the formatted and fully validated checkpoint dictionary!
            torch.save(ckpt, str(dest_path))
            
            # Reload dropdown values
            checkpoints = [p.name for p in self.hcfg.CHECKPOINT_DIR.glob("*_best.pth")]
            self.model_menu.configure(values=checkpoints)
            self.model_var.set(dest_path.name)
            
            # Success feedback
            num_classes = len(ckpt["class_names"])
            val_acc = ckpt.get("val_accuracy", ckpt.get("best_val_acc", 0.0))
            self.status_lbl.configure(
                text=f"✓ Successfully imported '{dest_path.name}' ({num_classes} classes, val_acc={val_acc:.2%})!",
                text_color=COLORS["success"]
            )
            
        except Exception as e:
            self.status_lbl.configure(text=f"❌ Import failed: {str(e)}", text_color=COLORS["error"])

    def destroy(self):
        self._stop_realtime()
        self._stop_gif()
        super().destroy()

    def _process_frame_telemetry(self, pred_class, all_probs):
        if not self.realtime_running:
            return
            
        # 1. Timeline Chrono-segmentation update
        self.session_timeline.append(pred_class)
        
        # 2. Cumulative Share tally updates (using full probabilities for mathematical smoothness)
        if all_probs:
            for cls, prob in all_probs.items():
                self.session_tally[cls] = self.session_tally.get(cls, 0.0) + prob
        else:
            self.session_tally[pred_class] = self.session_tally.get(pred_class, 0.0) + 1.0
            
        # 3. Trigger visual canvas redrawing and leaderboard bar sliding
        self._draw_timeline_canvas()
        self._update_session_leaderboard()

    def _get_class_color(self, class_name):
        if not hasattr(self, "class_colors"):
            self.class_colors = {}
        if class_name not in self.class_colors:
            palette = [
                "#4f46e5",  # Royal Indigo
                "#10b981",  # Emerald Green
                "#f59e0b",  # Amber Gold
                "#ef4444",  # Crimson Red
                "#3b82f6",  # Cobalt Blue
                "#ec4899",  # Slate Pink
                "#8b5cf6",  # Deep Violet
                "#14b8a6",  # Teal Cyan
                "#06b6d4",  # Cyan
                "#84cc16",  # Lime Green
            ]
            color_idx = len(self.class_colors) % len(palette)
            self.class_colors[class_name] = palette[color_idx]
        return self.class_colors[class_name]

    def _draw_timeline_canvas(self):
        if not hasattr(self, "timeline_canvas") or not self.session_timeline:
            return
            
        self.timeline_canvas.delete("all")
        w = self.timeline_canvas.winfo_width()
        h = self.timeline_canvas.winfo_height()
        
        if w < 10: w = 280
        if h < 5: h = 16
            
        N = len(self.session_timeline)
        
        # Compress contiguous identical predictions to optimize Tkinter drawing performance
        segments = []
        curr_class = self.session_timeline[0]
        start_idx = 0
        
        for i in range(1, N):
            cls = self.session_timeline[i]
            if cls != curr_class:
                segments.append((curr_class, start_idx, i - 1))
                curr_class = cls
                start_idx = i
        segments.append((curr_class, start_idx, N - 1))
        
        # Render color blocks onto Tkinter Canvas
        for cls, s_idx, e_idx in segments:
            x0 = s_idx * w / N
            x1 = (e_idx + 1) * w / N
            color = self._get_class_color(cls)
            self.timeline_canvas.create_rectangle(x0, 0, x1, h, fill=color, outline="")

    def _update_session_leaderboard(self):
        if not hasattr(self, "share_frame") or not self.session_tally:
            return
            
        total = sum(self.session_tally.values())
        if total <= 0:
            return
            
        sorted_shares = sorted(self.session_tally.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Pool & create Top-3 share widgets dynamically
        if not hasattr(self, "share_widgets") or len(self.share_widgets) != 3:
            for w in self.share_frame.winfo_children():
                w.destroy()
            self.share_widgets = []
            
            for i in range(3):
                row = ctk.CTkFrame(self.share_frame, fg_color="transparent")
                row.pack(fill="x", pady=2)
                
                # Dynamic visual dot indicator matching resolved class color
                indicator = ctk.CTkFrame(row, width=8, height=8, corner_radius=2)
                indicator.pack(side="left", padx=(2, 5))
                
                lbl = ctk.CTkLabel(row, text="-", font=("Segoe UI", 10), text_color=COLORS["text"],
                                   width=110, anchor="w")
                lbl.pack(side="left")
                
                bar = ctk.CTkProgressBar(row, width=110, height=8,
                                         progress_color=COLORS["accent"], fg_color=COLORS["progress_bg"])
                bar.pack(side="left", padx=5)
                bar.set(0.0)
                
                pct = ctk.CTkLabel(row, text="0.0%", font=("Segoe UI", 10),
                                   text_color=COLORS["text_dim"], width=40)
                pct.pack(side="left")
                
                self.share_widgets.append((indicator, lbl, bar, pct))
                
        # Populate Top-3 slots
        for i in range(3):
            indicator, lbl, bar, pct = self.share_widgets[i]
            if i < len(sorted_shares):
                cls, score = sorted_shares[i]
                share = score / total
                color = self._get_class_color(cls)
                
                # Dynamic matching theme styling
                indicator.configure(fg_color=color)
                lbl.configure(text=cls)
                bar.set(share)
                bar.configure(progress_color=color)
                pct.configure(text=f"{share:.1%}")
            else:
                # Clear empty slots cleanly
                indicator.configure(fg_color="transparent")
                lbl.configure(text="-")
                bar.set(0.0)
                bar.configure(progress_color=COLORS["accent"])
                pct.configure(text="0.0%")
