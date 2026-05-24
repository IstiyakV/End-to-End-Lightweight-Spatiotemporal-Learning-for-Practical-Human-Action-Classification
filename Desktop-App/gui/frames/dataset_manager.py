"""Dataset Manager — browse classes, preview, cache, download."""

import customtkinter as ctk
from pathlib import Path
import sys
import threading
import subprocess
import urllib.request
import shutil

from gui.settings import load_settings, save_setting
from gui.theme import COLORS

class DatasetManagerFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=COLORS["bg"], corner_radius=0)
        self.app = app
        self._ucf_downloading = False
        self._kin_downloading = False
        self._cache_process = None
        self._cache_paused = False
        self._cache_stop_flag = False
        # Kinetics download control events
        self._kin_pause_event = threading.Event()   # set = running, clear = paused
        self._kin_pause_event.set()  # start unpaused
        self._kin_stop_flag = False
        
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=30, pady=(25, 0))
        ctk.CTkLabel(header, text="Dataset Manager", font=("Segoe UI", 26, "bold"), text_color=COLORS["text"]).pack(anchor="w")
        ctk.CTkLabel(header, text="Browse, select locations, and cache datasets", font=("Segoe UI", 12), text_color=COLORS["text_dim"]).pack(anchor="w")

        # Tabview
        self.tabview = ctk.CTkTabview(self, fg_color=COLORS["card"], 
                                      segmented_button_selected_color=COLORS["accent"], 
                                      segmented_button_selected_hover_color="#7c6cf7",
                                      text_color=COLORS["text"])
        self.tabview.pack(fill="both", expand=True, padx=20, pady=(10, 20))
        
        self.tab_ucf = self.tabview.add("UCF-101")
        self.tab_kin = self.tabview.add("Kinetics-700")

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from har import config as hcfg
        self.hcfg = hcfg

        self._build_tab(self.tab_ucf, "ucf101", str(hcfg.UCF101_DIR))
        self._build_tab(self.tab_kin, "kinetics", str(hcfg.DATA_DIR))

    # ── layout helpers ────────────────────────────────────────────────────
    def _make_card(self, parent, num, title, subtitle):
        outer = ctk.CTkFrame(parent, fg_color=COLORS["card"],
                             border_color=COLORS["border"], border_width=1, corner_radius=10)
        outer.grid_columnconfigure(0, weight=1)
        hdr = ctk.CTkFrame(outer, fg_color="transparent")
        hdr.pack(fill="x", padx=14, pady=(10,4))
        ctk.CTkLabel(hdr, text=num, font=("Segoe UI",11,"bold"),
                     fg_color=COLORS["accent"], text_color="white",
                     width=24, height=24, corner_radius=12).pack(side="left", padx=(0,8))
        ctk.CTkLabel(hdr, text=title, font=("Segoe UI",13,"bold"),
                     text_color=COLORS["text"]).pack(side="left")
        ctk.CTkLabel(hdr, text=f"  —  {subtitle}", font=("Segoe UI",11),
                     text_color=COLORS["text_dim"]).pack(side="left")
        body = ctk.CTkFrame(outer, fg_color="transparent")
        body.pack(fill="x", padx=14, pady=(0,12))
        return outer, body

    def _path_row(self, parent, label, hint, var, browse_fn, extra=None):
        r = ctk.CTkFrame(parent, fg_color="transparent")
        r.pack(fill="x", pady=(6,0))
        ctk.CTkLabel(r, text=label, font=("Segoe UI",12,"bold"),
                     text_color=COLORS["text"], width=170, anchor="w").pack(side="left")
        ctk.CTkEntry(r, textvariable=var, width=310,
                     fg_color=COLORS["input_bg"], border_color=COLORS["border"],
                     font=("Segoe UI",11)).pack(side="left", padx=(0,6))
        ctk.CTkButton(r, text="Browse…", width=72,
                      fg_color=COLORS["border"], hover_color=COLORS["card_border"],
                      text_color=COLORS["text"], command=browse_fn).pack(side="left")
        if extra: extra(r)
        badge = ctk.CTkLabel(r, text="", font=("Segoe UI",10), text_color=COLORS["text_dim"])
        badge.pack(side="left", padx=(8,0))
        hr = ctk.CTkFrame(parent, fg_color="transparent")
        hr.pack(fill="x")
        ctk.CTkLabel(hr, text=f"    {hint}", font=("Segoe UI",10),
                     text_color=COLORS["text_dim"], anchor="w").pack(side="left", padx=(172,0))
        return badge

    def _build_tab(self, parent, ds_id, default_path):
        # 2-column split: left panel = scrollable controls, right = class explorer (always visible)
        parent.grid_columnconfigure(0, weight=0)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        settings         = load_settings()
        saved_ds_path    = settings.get(f"{ds_id}_ds_path", default_path)
        saved_cache_path = settings.get(f"{ds_id}_cache_path",
                          str(self.hcfg.PROJECT_ROOT / "cache" / ds_id))

        def _is_cloud(s):
            if not s: return False
            lo = s.lower().replace("\\", "/")
            if any(k in lo for k in ["my drive","google drive","onedrive","dropbox","sharepoint"]):
                return True
            try:
                import ctypes
                return ctypes.windll.kernel32.GetDriveTypeW(s[:3]) == 4
            except: return False

        # ── LEFT PANEL ───────────────────────────────────────────────────
        left_scroll = ctk.CTkScrollableFrame(parent, width=370,
                                             fg_color=COLORS["card"],
                                             corner_radius=0,
                                             scrollbar_button_color=COLORS["border"],
                                             scrollbar_button_hover_color=COLORS["card_border"])
        left_scroll.grid(row=0, column=0, sticky="nsew", padx=(10,0), pady=10)
        left_scroll.grid_columnconfigure(0, weight=1)

        def _section(title, subtitle=None):
            f = ctk.CTkFrame(left_scroll, fg_color=COLORS["input_bg"], corner_radius=8)
            f.pack(fill="x", pady=(0,8))
            ctk.CTkLabel(f, text=title, font=("Segoe UI",11,"bold"),
                         text_color=COLORS["text"]).pack(anchor="w", padx=10, pady=(8,0))
            if subtitle:
                ctk.CTkLabel(f, text=subtitle, font=("Segoe UI",9),
                             text_color=COLORS["text_dim"], wraplength=320,
                             justify="left").pack(anchor="w", padx=10, pady=(1,4))
            body = ctk.CTkFrame(f, fg_color="transparent")
            body.pack(fill="x", padx=10, pady=(0,10))
            return body

        def _field(parent, label, var, browse_fn, hint=None):
            ctk.CTkLabel(parent, text=label, font=("Segoe UI",10,"bold"),
                         text_color=COLORS["text_dim"], anchor="w").pack(fill="x", pady=(4,1))
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x")
            row.grid_columnconfigure(0, weight=1)
            ctk.CTkEntry(row, textvariable=var,
                         fg_color=COLORS["card"], border_color=COLORS["border"],
                         font=("Segoe UI",10), height=28
                         ).grid(row=0, column=0, sticky="ew", padx=(0,4))
            ctk.CTkButton(row, text="Browse", width=56, height=28,
                          fg_color=COLORS["border"], hover_color=COLORS["card_border"],
                          text_color=COLORS["text"], font=("Segoe UI",10),
                          command=browse_fn).grid(row=0, column=1)
            badge = ctk.CTkLabel(parent, text="", font=("Segoe UI",9),
                                 text_color=COLORS["text_dim"], anchor="w")
            badge.pack(fill="x")
            if hint:
                ctk.CTkLabel(parent, text=hint, font=("Segoe UI",9),
                             text_color=COLORS["text_dim"], wraplength=320,
                             justify="left").pack(anchor="w", pady=(0,2))
            return badge

        # SECTION: PATHS
        paths_body = _section("\u2460 Paths", "Where data lands and where it lives.")

        dl_root_var = None
        if ds_id == "kinetics":
            dl_root_var = ctk.StringVar(value=settings.get("kinetics_dl_root",""))
            def _browse_dl():
                d = ctk.filedialog.askdirectory(initialdir=dl_root_var.get() or ".")
                if d: dl_root_var.set(d); save_setting("kinetics_dl_root", d)
            dl_badge = _field(paths_body, "Download Directory (for ZIPs)",
                              dl_root_var, _browse_dl,
                              "Use a local drive, not Google Drive, for full speed.")
            warn_strip = ctk.CTkFrame(paths_body, fg_color="#3d2000", corner_radius=5)
            ctk.CTkLabel(warn_strip,
                text="\u26a0  Cloud drive detected. Use a local path (e.g. D:\\kinetics_temp) for ZIPs to avoid sync overhead.",
                font=("Segoe UI",9), text_color="#f39c12",
                wraplength=310, justify="left").pack(padx=8, pady=4)
            def _chk_dl(*_):
                p = dl_root_var.get()
                if _is_cloud(p):
                    warn_strip.pack(fill="x", pady=(2,0))
                    dl_badge.configure(text="\u26a0  Cloud drive — slower", text_color="#f39c12")
                elif p and Path(p).exists():
                    warn_strip.pack_forget()
                    dl_badge.configure(text="\u2705  Ready", text_color=COLORS["success"])
                elif p:
                    warn_strip.pack_forget()
                    dl_badge.configure(text="\u274c  Path not found", text_color=COLORS["error"])
                else:
                    warn_strip.pack_forget(); dl_badge.configure(text="")
            dl_root_var.trace_add("write", _chk_dl); _chk_dl()

        path_var = ctk.StringVar(value=saved_ds_path)
        def _browse_ds():
            d = ctk.filedialog.askdirectory(initialdir=path_var.get() or ".")
            if d: path_var.set(d); save_setting(f"{ds_id}_ds_path", d); load_classes()
        ds_badge = _field(paths_body, "Dataset Folder (extracted MP4s)",
                          path_var, _browse_ds,
                          "Can be Google Drive. Only reading, no ZIP sync.")
        def _chk_ds(*_):
            p = Path(path_var.get())
            if not path_var.get() or not p.exists():
                ds_badge.configure(text="\u274c  Not found", text_color=COLORS["error"])
            else:
                try: has = any(True for _ in p.iterdir())
                except: has = False
                ds_badge.configure(
                    text="\u2705  Dataset detected" if has else "\U0001f4c2  Empty folder",
                    text_color=COLORS["success"] if has else COLORS["text_dim"])
        path_var.trace_add("write", lambda *_: _chk_ds()); _chk_ds()

        # hidden stats_lbl for backend compat
        stats_lbl = ctk.CTkLabel(left_scroll, text="", font=("Segoe UI",9),
                                 text_color=COLORS["text_dim"])
        stats_lbl.pack(anchor="w", padx=10)

        # SECTION: GET DATASET
        if ds_id == "kinetics":
            get_body = _section("\u2461 Download", "Download ZIP parts from the internet.")
            
            # --- Mode Selection ---
            self.dl_mode_var = ctk.StringVar(value="all")
            mode_row = ctk.CTkFrame(get_body, fg_color="transparent")
            mode_row.pack(fill="x", pady=(0,6))
            ctk.CTkRadioButton(mode_row, text="All/Remaining Parts", variable=self.dl_mode_var, value="all", font=("Segoe UI",10)).pack(side="left", padx=(0,10))
            ctk.CTkRadioButton(mode_row, text="Single Part", variable=self.dl_mode_var, value="single", font=("Segoe UI",10)).pack(side="left")

            pr = ctk.CTkFrame(get_body, fg_color="transparent")
            pr.pack(fill="x", pady=(0,6))
            ctk.CTkLabel(pr, text="Parts:", font=("Segoe UI",10),
                         text_color=COLORS["text_dim"]).pack(side="left")
            self.kin_start_var = ctk.IntVar(value=1)
            self.kin_end_var   = ctk.IntVar(value=22)
            
            self.entry_start = ctk.CTkEntry(pr, textvariable=self.kin_start_var, width=42, height=26,
                         fg_color=COLORS["card"], border_color=COLORS["border"], font=("Segoe UI",10))
            self.entry_start.pack(side="left", padx=(6,3))
            
            self.lbl_to = ctk.CTkLabel(pr, text="to", font=("Segoe UI",10), text_color=COLORS["text_dim"])
            self.lbl_to.pack(side="left")
            
            self.entry_end = ctk.CTkEntry(pr, textvariable=self.kin_end_var, width=42, height=26,
                         fg_color=COLORS["card"], border_color=COLORS["border"], font=("Segoe UI",10))
            self.entry_end.pack(side="left", padx=3)
            
            self.lbl_of = ctk.CTkLabel(pr, text="of 22  (~19 GB each)", font=("Segoe UI",9), text_color=COLORS["text_dim"])
            self.lbl_of.pack(side="left", padx=(4,0))
            
            def _on_mode_change(*_):
                if self.dl_mode_var.get() == "all":
                    self.lbl_to.pack(side="left")
                    self.entry_end.pack(side="left", padx=3)
                    self.lbl_of.configure(text="of 22  (~19 GB each)")
                    self.kin_start_var.set(1)
                    self.kin_end_var.set(22)
                else:
                    self.lbl_to.pack_forget()
                    self.entry_end.pack_forget()
                    self.lbl_of.configure(text="of 22")
                if hasattr(self, "_update_kin_dl_btn"):
                    self._update_kin_dl_btn()
            self.dl_mode_var.trace_add("write", _on_mode_change)

            def _update_kin_dl_btn():
                completed = 0
                try:
                    cf = Path(dl_root_var.get())/"completed_parts.txt"
                    if cf.exists():
                        completed = sum(1 for l in open(cf) if l.strip())
                except: pass
                if self.dl_mode_var.get() == "single":
                    self.kin_dl_btn.configure(text="Download Part")
                else:
                    if completed == 0:
                        self.kin_dl_btn.configure(text="Start Download")
                    elif completed < 22:
                        self.kin_dl_btn.configure(text=f"Download rest of the parts ({22-completed} left)")
                    else:
                        self.kin_dl_btn.configure(text="Redownload Dataset")
            self._update_kin_dl_btn = _update_kin_dl_btn

            def _start_dl():
                start_p = self.kin_start_var.get()
                end_p = self.kin_end_var.get() if self.dl_mode_var.get() == "all" else start_p
                self._download_kinetics(dl_root_var.get(), stats_lbl, start_p, end_p)

            self.kin_dl_btn = ctk.CTkButton(get_body, text="Start Download", height=30,
                fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                font=("Segoe UI",11,"bold"), command=_start_dl)
            self.kin_dl_btn.pack(fill="x", pady=(0,3))

            # Control row - NOT packed initially
            self.kin_ctrl_row = ctk.CTkFrame(get_body, fg_color="transparent")
            self.kin_ctrl_row.grid_columnconfigure(0, weight=1)
            self.kin_ctrl_row.grid_columnconfigure(1, weight=1)
            
            self.kin_pause_btn = ctk.CTkButton(self.kin_ctrl_row, text="Pause", height=28,
                fg_color="#e58e26", hover_color="#f39c12", font=("Segoe UI",10), command=self._toggle_kin_pause)
            self.kin_pause_btn.grid(row=0, column=0, padx=(0,3), sticky="ew")
            
            self.kin_stop_btn = ctk.CTkButton(self.kin_ctrl_row, text="Stop", height=28,
                fg_color=COLORS["error"], hover_color="#c0392b", font=("Segoe UI",10), command=self._stop_kin_download)
            self.kin_stop_btn.grid(row=0, column=1, padx=(3,0), sticky="ew")
            
            self.kin_extract_btn = ctk.CTkButton(get_body, text="Extract Found Parts",
                height=28, fg_color="#e58e26", hover_color="#f39c12", font=("Segoe UI",10),
                command=lambda: self._extract_only_kinetics(dl_root_var.get(), path_var.get(), stats_lbl))

            self.kin_dl_progress = ctk.CTkProgressBar(get_body, height=8, progress_color=COLORS["success"])
            self.kin_dl_progress.set(0)
            self.kin_dl_lbl = ctk.CTkLabel(get_body, text="", font=("Segoe UI",9), text_color=COLORS["text_dim"], anchor="w")
            
            self.kin_ex_row = ctk.CTkFrame(get_body, fg_color="transparent")
            self.kin_ex_progress = ctk.CTkProgressBar(self.kin_ex_row, height=6, progress_color="#27ae60")
            self.kin_ex_progress.set(0)
            self.kin_ex_lbl = ctk.CTkLabel(self.kin_ex_row, text="", font=("Segoe UI",9), text_color="#27ae60", anchor="w")
            
            hint_f = ctk.CTkFrame(get_body, fg_color="transparent")
            hint_f.pack(fill="x", pady=(6,0))
            ctk.CTkFrame(hint_f, fg_color=COLORS["border"], height=1).pack(fill="x", pady=(0,6))
            ctk.CTkLabel(hint_f, text="Already have ZIPs on disk? Place them in temp_zip/ inside your Download Directory and the Extract button will appear above.",
                font=("Segoe UI",9), text_color=COLORS["text_dim"], wraplength=320, justify="left").pack(anchor="w")

        else:
            get_body = _section("\u2461 Download", "UCF-101 is ~6.5 GB.")
            self.ucf_dl_btn = ctk.CTkButton(get_body, text="Download Dataset", height=30,
                fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                font=("Segoe UI",11,"bold"),
                command=lambda: self._download_ucf(path_var.get(), stats_lbl))
            self.ucf_dl_btn.pack(fill="x", pady=(0,4))
            self.ucf_extract_btn = ctk.CTkButton(get_body, text="Use Local RAR File...",
                height=28, fg_color="#e58e26", hover_color="#f39c12", font=("Segoe UI",10),
                command=lambda: self._prompt_extract_ucf(path_var.get(), stats_lbl))
            self.ucf_extract_btn.pack(fill="x", pady=(0,4))
            ctk.CTkLabel(get_body,
                text="Already have UCF101.rar? Click above to extract without re-downloading.",
                font=("Segoe UI",9), text_color=COLORS["text_dim"],
                wraplength=320, justify="left").pack(anchor="w")
            self.ucf_dl_progress = ctk.CTkProgressBar(get_body, height=8,
                                                       progress_color=COLORS["success"])
            self.ucf_dl_progress.set(0)
            self.ucf_dl_lbl = ctk.CTkLabel(get_body, text="", font=("Segoe UI",9),
                                           text_color=COLORS["text_dim"], anchor="w")

        # SECTION: CACHE
        enable_cache_var = ctk.BooleanVar(value=(ds_id != "kinetics"))
        cache_path_var   = ctk.StringVar(value=saved_cache_path)
        cache_body = _section("\u2462 Build Cache",
                              "Pre-process videos into .npy arrays for 10-50x faster training.")
        sw_row = ctk.CTkFrame(cache_body, fg_color="transparent")
        sw_row.pack(fill="x", pady=(0,4))
        ctk.CTkSwitch(sw_row, text="Enable caching", variable=enable_cache_var,
                      font=("Segoe UI",10,"bold"), progress_color=COLORS["success"],
                      command=lambda: _tog_cache()).pack(side="left")

        cache_cfg = ctk.CTkFrame(cache_body, fg_color="transparent")
        ctk.CTkLabel(cache_cfg, text="Output folder:", font=("Segoe UI",10),
                     text_color=COLORS["text_dim"], anchor="w").pack(fill="x")
        cp_row = ctk.CTkFrame(cache_cfg, fg_color="transparent")
        cp_row.pack(fill="x", pady=(1,6))
        cp_row.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(cp_row, textvariable=cache_path_var,
                     fg_color=COLORS["card"], border_color=COLORS["border"],
                     font=("Segoe UI",10), height=28
                     ).grid(row=0, column=0, sticky="ew", padx=(0,4))
        def _browse_cache():
            d = ctk.filedialog.askdirectory(initialdir=cache_path_var.get() or ".")
            if d: cache_path_var.set(d); save_setting(f"{ds_id}_cache_path", d)
        ctk.CTkButton(cp_row, text="Browse", width=56, height=28,
                      fg_color=COLORS["border"], hover_color=COLORS["card_border"],
                      text_color=COLORS["text"], font=("Segoe UI",10),
                      command=_browse_cache).grid(row=0, column=1)
        cache_progress = ctk.CTkProgressBar(cache_cfg, height=8, progress_color=COLORS["success"])
        cache_progress.set(0)
        
        cache_btn_row = ctk.CTkFrame(cache_cfg, fg_color="transparent")
        cache_btn_row.pack(fill="x", pady=(0,3))
        cache_btn_row.grid_columnconfigure(0, weight=1)
        cache_btn_row.grid_columnconfigure(1, weight=0)

        cache_btn_primary = ctk.CTkButton(cache_btn_row, text="Build .npy Cache", height=30,
            fg_color=COLORS["accent"], hover_color="#7c6cf7", font=("Segoe UI",11,"bold"),
            command=lambda: self._cache_dataset(
                cache_path_var.get(), path_var.get(), stats_lbl, cache_btn_primary, cache_btn_rebuild, cache_pause_btn, cache_stop_btn, cache_progress, rebuild=False))
        cache_btn_primary.grid(row=0, column=0, sticky="ew", padx=(0,4))

        cache_btn_rebuild = ctk.CTkButton(cache_btn_row, text="Rebuild Cache", height=30, width=90,
            fg_color="transparent", border_color=COLORS["error"], border_width=1,
            text_color=COLORS["error"], hover_color="#c0392b", font=("Segoe UI",10,"bold"),
            command=lambda: self._cache_dataset(
                cache_path_var.get(), path_var.get(), stats_lbl, cache_btn_primary, cache_btn_rebuild, cache_pause_btn, cache_stop_btn, cache_progress, rebuild=True))
        
        def _update_cache_btns(*_):
            try:
                p = Path(cache_path_var.get())
                if p.exists() and any(p.iterdir()):
                    cache_btn_primary.configure(text="Recheck & Rebuild Cache")
                    cache_btn_rebuild.grid(row=0, column=1, sticky="ew")
                else:
                    cache_btn_primary.configure(text="Build .npy Cache")
                    cache_btn_rebuild.grid_forget()
            except: pass
        cache_path_var.trace_add("write", lambda *_: _update_cache_btns())
        self.after(500, _update_cache_btns)
        cc_row = ctk.CTkFrame(cache_cfg, fg_color="transparent")
        cc_row.pack(fill="x")
        cc_row.grid_columnconfigure(0, weight=1)
        cc_row.grid_columnconfigure(1, weight=1)
        cache_pause_btn = ctk.CTkButton(cc_row, text="Pause", height=28,
            fg_color="#e58e26", hover_color="#f39c12", font=("Segoe UI",10),
            command=lambda: self._toggle_cache_pause(cache_pause_btn, stats_lbl))
        cache_stop_btn = ctk.CTkButton(cc_row, text="Stop", height=28,
            fg_color=COLORS["error"], hover_color="#c0392b", font=("Segoe UI",10),
            command=lambda: self._stop_cache(
                cache_btn_primary, cache_btn_rebuild, cache_pause_btn, cache_stop_btn, cache_progress, stats_lbl))

        def _tog_cache():
            if enable_cache_var.get():
                cache_cfg.pack(fill="x", pady=(0,0))
            else:
                cache_cfg.pack_forget()
                cache_progress.pack_forget()
                cache_pause_btn.grid_forget()
                cache_stop_btn.grid_forget()
        _tog_cache()

        # ── RIGHT PANEL: CLASS EXPLORER ───────────────────────────────────
        right = ctk.CTkFrame(parent, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=10)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        top_bar = ctk.CTkFrame(right, fg_color="transparent")
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0,6))
        top_bar.grid_columnconfigure(0, weight=1)

        search_var = ctk.StringVar()
        ctk.CTkEntry(top_bar, textvariable=search_var,
                     placeholder_text="Search classes...",
                     fg_color=COLORS["input_bg"], border_color=COLORS["border"],
                     font=("Segoe UI",11)
                     ).grid(row=0, column=0, sticky="ew", padx=(0,8))

        stats_bar = ctk.CTkFrame(top_bar, fg_color="transparent")
        stats_bar.grid(row=0, column=1)

        def _pill(container, text, color):
            f = ctk.CTkFrame(container, fg_color=COLORS["card"],
                             border_color=COLORS["border"], border_width=1, corner_radius=6)
            f.pack(side="left", padx=3)
            ctk.CTkLabel(f, text=text, font=("Segoe UI",10),
                         text_color=color).pack(padx=8, pady=4)
            return f

        cls_pill  = _pill(stats_bar, "Classes: -", COLORS["text_dim"])
        vid_pill  = _pill(stats_bar, "Videos: -", COLORS["text_dim"])
        size_pill = _pill(stats_bar, "Size: -", COLORS["text_dim"])

        scroll = ctk.CTkScrollableFrame(right, fg_color=COLORS["input_bg"], corner_radius=8)
        scroll.grid(row=1, column=0, sticky="nsew")
        
        pagination_bar = ctk.CTkFrame(right, fg_color="transparent")
        pagination_bar.grid(row=2, column=0, sticky="ew", pady=(4,0))

        all_rows = []



        def load_classes(*_):
            save_setting(f"{ds_id}_ds_path", path_var.get())
            for w in scroll.winfo_children(): w.destroy()
            all_rows.clear()
            p = Path(path_var.get())
            if not p.exists() or not p.is_dir():
                stats_lbl.configure(text="Directory not found.", text_color=COLORS["error"])
                return
            if ds_id == "ucf101":   self.ucf_load_classes = load_classes
            if ds_id == "kinetics": self.kin_load_classes  = load_classes

            scan_results = None; scan_error = None; scan_done = False

            def scan_thread():
                nonlocal scan_results, scan_error, scan_done
                try:
                    classes = sorted([d.name for d in p.iterdir()
                                      if d.is_dir() and not d.name.startswith(".")])
                    results = []; total_vids = 0
                    for cls in classes:
                        n = sum(1 for f in (p/cls).iterdir()
                                if f.suffix.lower() in {".mp4",".avi",".mov",".mkv"})
                        total_vids += n; results.append((cls, n))
                    scan_results = (results, total_vids)
                except Exception as e: scan_error = e
                finally: scan_done = True

            def check_thread():
                if not scan_done: self.after(100, check_thread); return
                if scan_error:
                    stats_lbl.configure(text=f"Error: {scan_error}", text_color=COLORS["error"]); return
                res, tv = scan_results
                if ds_id == "ucf101" and hasattr(self,"ucf_dl_btn"):
                    self.ucf_dl_btn.configure(text="Redownload Dataset" if res else "Download Dataset")
                if ds_id == "kinetics" and hasattr(self,"kin_dl_btn"):
                    if hasattr(self, "_update_kin_dl_btn"):
                        self._update_kin_dl_btn()
                    try:
                        tz = Path(dl_root_var.get())/"temp_zip"
                        orphans = list(tz.glob("Kinetics700_part_*.zip")) if tz.exists() else []
                        if orphans:
                            self.kin_extract_btn.configure(text=f"Extract Found Parts ({len(orphans)})")
                            self.kin_extract_btn.pack(fill="x", pady=(4,0))
                        else:
                            self.kin_extract_btn.pack_forget()
                    except: pass
                if not res:
                    stats_lbl.configure(text="No class folders found.", text_color=COLORS["warning"]); return
                build_ui(res, tv)

            def build_ui(results, total_vids):
                PAGE_SIZE = 50
                current_page = [0]
                filtered_results = results.copy()
                total_pages = [ (len(filtered_results) + PAGE_SIZE - 1) // PAGE_SIZE or 1 ]

                def render_page(page_idx):
                    for w in scroll.winfo_children(): w.destroy()
                    all_rows.clear()
                    
                    start_idx = page_idx * PAGE_SIZE
                    page_items = filtered_results[start_idx : start_idx + PAGE_SIZE]
                    
                    for cls, n_vids in page_items:
                        fr = ctk.CTkFrame(scroll, fg_color="transparent", height=30)
                        fr.pack(fill="x", pady=2)
                        all_rows.append((fr, cls))
                        ctk.CTkLabel(fr, text=cls, font=("Segoe UI",12),
                                     text_color=COLORS["text"], width=250, anchor="w"
                                     ).pack(side="left", padx=10)
                        ctk.CTkLabel(fr, text=f"{n_vids} videos", font=("Segoe UI",11),
                                     text_color=COLORS["text_dim"], width=80
                                     ).pack(side="right", padx=10)
                        bar = ctk.CTkProgressBar(fr, width=100, height=4,
                                                 progress_color=COLORS["accent"],
                                                 fg_color=COLORS["progress_bg"])
                        bar.pack(side="right", padx=5)
                        bar.set(min(n_vids/150.0, 1.0))

                    page_lbl.configure(text=f"Page {page_idx+1} of {total_pages[0]}")
                    prev_btn.configure(state="normal" if page_idx > 0 else "disabled")
                    next_btn.configure(state="normal" if page_idx < total_pages[0] - 1 else "disabled")

                def go_prev():
                    if current_page[0] > 0:
                        current_page[0] -= 1
                        render_page(current_page[0])

                def go_next():
                    if current_page[0] < total_pages[0] - 1:
                        current_page[0] += 1
                        render_page(current_page[0])

                for w in pagination_bar.winfo_children(): w.destroy()
                prev_btn = ctk.CTkButton(pagination_bar, text="< Prev", width=60, height=24, fg_color=COLORS["border"], hover_color=COLORS["card_border"], text_color=COLORS["text"], command=go_prev)
                prev_btn.pack(side="left")
                page_lbl = ctk.CTkLabel(pagination_bar, text="", font=("Segoe UI", 10), text_color=COLORS["text_dim"])
                page_lbl.pack(side="left", expand=True)
                next_btn = ctk.CTkButton(pagination_bar, text="Next >", width=60, height=24, fg_color=COLORS["border"], hover_color=COLORS["card_border"], text_color=COLORS["text"], command=go_next)
                next_btn.pack(side="right")

                for w in cls_pill.winfo_children():  w.destroy()
                for w in vid_pill.winfo_children():  w.destroy()
                for w in size_pill.winfo_children(): w.destroy()
                ctk.CTkLabel(cls_pill, text=f"{len(results)} classes", font=("Segoe UI",10), text_color=COLORS["text"]).pack(padx=8,pady=4)
                ctk.CTkLabel(vid_pill, text=f"{total_vids:,} videos", font=("Segoe UI",10), text_color=COLORS["text"]).pack(padx=8,pady=4)
                ctk.CTkLabel(size_pill, text="Calculating...", font=("Segoe UI",10), text_color=COLORS["text_dim"]).pack(padx=8,pady=4)

                def size_thread():
                    try:
                        total_bytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                        s = (f"{total_bytes/1024**3:.2f} GB" if total_bytes > 1024**3 else f"{total_bytes/1024**2:.2f} MB")
                        def upd():
                            for w in size_pill.winfo_children(): w.destroy()
                            ctk.CTkLabel(size_pill, text=s, font=("Segoe UI",10), text_color=COLORS["success"]).pack(padx=8,pady=4)
                        self.after(0, upd)
                    except: pass
                threading.Thread(target=size_thread, daemon=True).start()

                def _filter(*_):
                    q = search_var.get().lower()
                    nonlocal filtered_results
                    if not q:
                        filtered_results = results.copy()
                    else:
                        filtered_results = [r for r in results if q in r[0].lower()]
                    total_pages[0] = (len(filtered_results) + PAGE_SIZE - 1) // PAGE_SIZE or 1
                    current_page[0] = 0
                    render_page(0)
                
                # IMPORTANT: We only want one trace active at a time to avoid memory leaks or multiple calls
                if hasattr(self, '_current_search_trace'):
                    try: search_var.trace_remove("write", self._current_search_trace)
                    except: pass
                self._current_search_trace = search_var.trace_add("write", _filter)

                render_page(0)

            threading.Thread(target=scan_thread, daemon=True).start()
            check_thread()

        load_classes()


    def _cache_dataset(self, target_cache_dir, raw_dataset_path, stats_lbl, cache_btn_primary, cache_btn_rebuild, cache_pause_btn, cache_stop_btn, cache_progress, rebuild=False):
        if self._cache_process is not None:
            return  # already running
            
        p = Path(raw_dataset_path)
        if not p.exists():
            stats_lbl.configure(text="Dataset folder not found.", text_color=COLORS["error"])
            return
            
        cache_dir = Path(target_cache_dir)
        if not target_cache_dir:
            stats_lbl.configure(text="Please set a Cache Output directory first.", text_color=COLORS["error"])
            return
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset flags
        self._cache_paused = False
        self._cache_stop_flag = False
        
        # Show controls
        cache_btn_primary.configure(state="disabled", text="▶ Running...")
        if hasattr(self, 'cache_btn_rebuild'):
            cache_btn_rebuild.configure(state="disabled")
        cache_pause_btn.configure(text="Pause")
        cache_pause_btn.grid(row=0, column=0, padx=(0,3), sticky="ew")
        cache_stop_btn.grid(row=0, column=1, padx=(3,0), sticky="ew")
        cache_progress.pack(fill="x", pady=(6,0))
        cache_progress.set(0)
        stats_lbl.configure(text=f"Initializing cache build...", text_color=COLORS["warning"])
        
        cache_done = False
        cache_error = None
        cache_prog_val = 0.0
        cache_status_text = ""
        
        def run():
            nonlocal cache_done, cache_error, cache_prog_val, cache_status_text
            try:
                import signal, os
                cmd = [
                    sys.executable, "scripts/preprocess_cache.py",
                    "--data-dir", str(p),
                    "--cache-dir", str(cache_dir),
                    "--workers", "4",
                    "--gui-mode"
                ]
                if rebuild:
                    cmd.append("--rebuild")
                
                proc = subprocess.Popen(cmd, cwd=str(self.hcfg.PROJECT_ROOT),
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                self._cache_process = proc
                
                for line in iter(proc.stdout.readline, ''):
                    # Check for stop
                    if self._cache_stop_flag:
                        proc.terminate()
                        try: proc.wait(timeout=3)
                        except: proc.kill()
                        cache_error = "STOPPED"
                        break
                    
                    # Check for pause — suspend/resume the process
                    if self._cache_paused:
                        try:
                            import signal as _s
                            if hasattr(_s, 'SIGSTOP'):
                                proc.send_signal(_s.SIGSTOP)
                            else:
                                # Windows: use taskkill trick or just sleep the reader thread
                                pass
                        except Exception:
                            pass
                        # Block the reader thread while paused
                        while self._cache_paused and not self._cache_stop_flag:
                            import time; time.sleep(0.5)
                        if not self._cache_stop_flag:
                            try:
                                import signal as _s
                                if hasattr(_s, 'SIGCONT'):
                                    proc.send_signal(_s.SIGCONT)
                            except Exception:
                                pass
                    
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("PROGRESS:"):
                        parts = line.split(":")[1].split("/")
                        if len(parts) == 2:
                            c, t = int(parts[0]), int(parts[1])
                            cache_prog_val = c / t
                            cache_status_text = f"Caching: {c}/{t} ({cache_prog_val*100:.1f}%) — {'PAUSED' if self._cache_paused else 'running'}"
                    elif any(k in line for k in ("Found", "Processing", "Done", "Skipped")):
                        cache_status_text = line
                        
                proc.stdout.close()
                if not self._cache_stop_flag:
                    ret = proc.wait()
                    if ret != 0:
                        cache_error = f"Process exited with code {ret}"
            except Exception as e:
                cache_error = str(e)
            finally:
                self._cache_process = None
                self._cache_paused = False
                cache_done = True
            
        def check_cache():
            if cache_error == "STOPPED":
                stats_lbl.configure(text="Cache stopped. Progress saved — resume anytime.", text_color=COLORS["warning"])
                cache_btn_primary.configure(state="normal", text="▶ Resume Cache")
                cache_btn_rebuild.configure(state="normal")
                cache_pause_btn.grid_forget()
                cache_stop_btn.grid_forget()
                cache_progress.pack_forget()
            elif cache_error:
                stats_lbl.configure(text=f"Error: {cache_error}", text_color=COLORS["error"])
                cache_btn_primary.configure(state="normal", text="▶ Build .npy Cache")
                cache_btn_rebuild.configure(state="normal")
                cache_pause_btn.grid_forget()
                cache_stop_btn.grid_forget()
                cache_progress.pack_forget()
            elif cache_done:
                stats_lbl.configure(text=f"✅ Cache complete! Saved to {cache_dir}", text_color=COLORS["success"])
                cache_btn_primary.configure(state="normal", text="Recheck & Rebuild Cache")
                cache_btn_rebuild.grid(row=0, column=1, sticky="ew")
                cache_btn_rebuild.configure(state="normal")
                cache_pause_btn.grid_forget()
                cache_stop_btn.grid_forget()
                cache_progress.pack_forget()
            else:
                if cache_status_text:
                    stats_lbl.configure(text=cache_status_text)
                cache_progress.set(cache_prog_val)
                self.after(200, check_cache)
            
        threading.Thread(target=run, daemon=True).start()
        check_cache()

    def _toggle_cache_pause(self, cache_pause_btn, stats_lbl):
        if self._cache_process is None:
            return
        self._cache_paused = not self._cache_paused
        if self._cache_paused:
            cache_pause_btn.configure(text="▶ Resume")
            stats_lbl.configure(text="⏸ Cache paused. Click Resume to continue.", text_color=COLORS["warning"])
        else:
            cache_pause_btn.configure(text="⏸ Pause")
            stats_lbl.configure(text="Resuming cache...", text_color=COLORS["text_dim"])

    def _stop_cache(self, cache_btn, cache_pause_btn, cache_stop_btn, cache_progress, stats_lbl):
        if self._cache_process is None:
            return
        self._cache_paused = False   # unblock reader thread if paused
        self._cache_stop_flag = True  # signal reader loop to terminate process

    def _download_ucf(self, target_dir_str, stats_lbl):
        if self._ucf_downloading:
            return
        p = Path(target_dir_str)
        if not target_dir_str or not p.exists() or not p.is_dir():
            stats_lbl.configure(text="Please set a valid target directory first using Browse...", text_color=COLORS["error"])
            return

        # Prepare UI
        self._ucf_downloading = True
        self.ucf_dl_btn.configure(state="disabled")
        self.ucf_dl_progress.pack(fill="x", padx=12, pady=(6,0))
        self.ucf_dl_lbl.pack(fill="x", padx=12)
        self.ucf_dl_progress.set(0)
        self.ucf_dl_lbl.configure(text="Starting download...", text_color=COLORS["text"])
        
        url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
        temp_dir = p / ".ucf_temp"
        temp_dir.mkdir(exist_ok=True)
        rar_path = temp_dir / "UCF101.rar"
        extracted_dir = temp_dir / "extracted"
        
        dl_progress_val = 0.0
        dl_status_text = ""
        dl_done = False
        dl_error = None
        
        def download_thread():
            nonlocal dl_progress_val, dl_status_text, dl_done, dl_error
            try:
                def reporthook(block_num, block_size, total_size):
                    nonlocal dl_progress_val, dl_status_text
                    if total_size > 0:
                        downloaded = block_num * block_size
                        dl_progress_val = min(downloaded / total_size, 1.0)
                        dl_status_text = f"Downloading: {downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB"
                
                urllib.request.urlretrieve(url, str(rar_path), reporthook)
                
                # Download done, now extract
                dl_progress_val = 1.0
                dl_status_text = "Extracting RAR archive (this may take a while)..."
                
                extracted_dir.mkdir(exist_ok=True)
                
                # Run 7-zip
                result = subprocess.run([
                    "C:\\Program Files\\7-Zip\\7z.exe", 
                    "x", str(rar_path), f"-o{extracted_dir}", "-y"
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"7z extraction failed: {result.stderr}")
                    
                dl_status_text = "Moving files to target directory..."
                dl_progress_val = 0.5
                
                # The RAR contains a 'UCF-101' folder
                inner_dir = extracted_dir / "UCF-101"
                if inner_dir.exists() and inner_dir.is_dir():
                    for item in inner_dir.iterdir():
                        if item.is_dir():
                            dest = p / item.name
                            if dest.exists():
                                shutil.rmtree(dest) # Overwrite if exists
                            shutil.move(str(item), str(p))
                
                dl_status_text = "Cleaning up temporary files..."
                dl_progress_val = 0.9
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                dl_progress_val = 1.0
                dl_status_text = "Dataset successfully installed!"
                
            except Exception as e:
                dl_error = str(e)
            finally:
                dl_done = True

        def check_ui():
            if dl_error:
                self._ucf_downloading = False
                self.ucf_dl_btn.configure(state="normal")
                self.ucf_dl_lbl.configure(text=f"Error: {dl_error}", text_color=COLORS["error"])
                self.ucf_dl_progress.pack_forget()
            elif dl_done:
                self._ucf_downloading = False
                self.ucf_dl_btn.configure(state="normal")
                self.ucf_dl_lbl.configure(text=dl_status_text, text_color=COLORS["success"])
                self.ucf_dl_progress.pack_forget()
                
                if hasattr(self, "ucf_load_classes"):
                    self.ucf_load_classes()
            else:
                self.ucf_dl_progress.set(dl_progress_val)
                self.ucf_dl_lbl.configure(text=dl_status_text)
                self.after(100, check_ui)

        threading.Thread(target=download_thread, daemon=True).start()
        check_ui()

    def _prompt_extract_ucf(self, target_dir_str, stats_lbl):
        p = Path(target_dir_str)
        if not target_dir_str or not p.exists() or not p.is_dir():
            stats_lbl.configure(text="Please set a valid Dataset MP4 Folder first.", text_color=COLORS["error"])
            return
            
        rar_file = ctk.filedialog.askopenfilename(
            title="Select Local UCF101.rar",
            filetypes=[("RAR archives", "*.rar"), ("All files", "*.*")]
        )
        if not rar_file:
            return
            
        self._extract_only_ucf(target_dir_str, stats_lbl, rar_file)

    def _extract_only_ucf(self, target_dir_str, stats_lbl, rar_path_str):
        if self._ucf_downloading:
            return
            
        p = Path(target_dir_str)
        temp_dir = p / ".ucf_temp"
        rar_path = Path(rar_path_str)
        
        if not p.exists() or not rar_path.exists():
            return
            
        self._ucf_downloading = True
        self.ucf_dl_btn.configure(state="disabled")
        self.ucf_extract_btn.configure(state="disabled")
        self.ucf_dl_progress.pack(fill="x", padx=12, pady=(6,0))
        self.ucf_dl_lbl.pack(fill="x", padx=12)
        self.ucf_dl_progress.set(0)
        self.ucf_dl_lbl.configure(text="Preparing to extract UCF101...", text_color=COLORS["text"])
        
        extracted_dir = temp_dir / "extracted"
        
        dl_progress_val = 0.0
        dl_status_text = ""
        dl_done = False
        dl_error = None
        
        def extract_thread():
            nonlocal dl_progress_val, dl_status_text, dl_done, dl_error
            try:
                dl_progress_val = 1.0
                dl_status_text = "Extracting RAR archive (this may take a while)..."
                
                extracted_dir.mkdir(exist_ok=True)
                
                result = subprocess.run([
                    "C:\\Program Files\\7-Zip\\7z.exe", 
                    "x", str(rar_path), f"-o{extracted_dir}", "-y"
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"7z extraction failed: {result.stderr}")
                    
                dl_status_text = "Moving files to target directory..."
                dl_progress_val = 0.5
                
                inner_dir = extracted_dir / "UCF-101"
                if inner_dir.exists() and inner_dir.is_dir():
                    for item in inner_dir.iterdir():
                        if item.is_dir():
                            dest = p / item.name
                            if dest.exists():
                                shutil.rmtree(dest)
                            shutil.move(str(item), str(p))
                
                dl_status_text = "Cleaning up temporary files..."
                dl_progress_val = 0.9
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                dl_progress_val = 1.0
                dl_status_text = "Dataset successfully installed from local file!"
            except Exception as e:
                dl_error = str(e)
            finally:
                dl_done = True
                
        def check_ui():
            if dl_error:
                self._ucf_downloading = False
                self.ucf_dl_btn.configure(state="normal")
                self.ucf_extract_btn.configure(state="normal")
                self.ucf_dl_lbl.configure(text=f"Error: {dl_error}", text_color=COLORS["error"])
                self.ucf_dl_progress.pack_forget()
            elif dl_done:
                self._ucf_downloading = False
                self.ucf_dl_btn.configure(state="normal")
                self.ucf_extract_btn.configure(state="normal")
                self.ucf_dl_lbl.configure(text=dl_status_text, text_color=COLORS["success"])
                self.ucf_dl_progress.pack_forget()
                
                if hasattr(self, "ucf_load_classes"):
                    self.ucf_load_classes()
            else:
                self.ucf_dl_progress.set(dl_progress_val)
                self.ucf_dl_lbl.configure(text=dl_status_text)
                self.after(100, check_ui)
                
        threading.Thread(target=extract_thread, daemon=True).start()
        check_ui()

    def _download_kinetics(self, target_dir_str, stats_lbl, start_part=1, end_part=22):
        if self._kin_downloading:
            return
        p = Path(target_dir_str)
        if not target_dir_str or not p.exists() or not p.is_dir():
            stats_lbl.configure(text="Please set a valid Download Directory first.", text_color=COLORS["error"])
            return

        # Clamp part range
        start_part = max(1, min(start_part, 22))
        end_part   = max(start_part, min(end_part, 22))

        # Reset control flags
        self._kin_pause_event.set()  # ensure unpaused
        self._kin_stop_flag = False
        self._kin_downloading = True

        # Show UI
        self.kin_dl_btn.configure(state="disabled")
        self.kin_ctrl_row.pack(fill="x", pady=(0,4))
        self.kin_pause_btn.configure(text="Pause")
        self.kin_pause_btn.grid(row=0, column=0, padx=(0,3), sticky="ew")
        self.kin_stop_btn.grid(row=0, column=1, padx=(3,0), sticky="ew")
        self.kin_dl_progress.pack(fill="x", padx=12, pady=(6,0))
        self.kin_dl_lbl.pack(fill="x", padx=12)
        self.kin_dl_progress.set(0)
        self.kin_dl_lbl.configure(text="Checking resume state...", text_color=COLORS["text"])

        temp_dir       = p / "temp_zip"
        temp_dir.mkdir(exist_ok=True)
        completed_file = p / "completed_parts.txt"
        completed_file.touch(exist_ok=True)

        dl_progress_val  = 0.0
        dl_status_text   = ""
        dl_done          = False
        dl_error         = None
        # Shared speed/progress state (written by chunk threads, read by UI)
        _speed_bps       = [0.0]   # bytes/sec current
        _downloaded_now  = [0]     # bytes downloaded for current part
        _total_now       = [0]     # total bytes for current part

        # ── IDM-style chunked download helper ──────────────────────────
        NUM_CONNECTIONS = 8  # parallel connections per file (like IDM)

        def _download_chunk(url, dest_path, byte_start, byte_end, buf_path, pause_event, stop_flag_ref):
            """Download a single byte range into a temp file."""
            import time
            headers = {"Range": f"bytes={byte_start}-{byte_end}"}
            req = urllib.request.Request(url, headers=headers)
            CHUNK = 256 * 1024  # 256 KB read buffer
            with urllib.request.urlopen(req) as resp, open(buf_path, "wb") as f:
                t0 = time.time()
                while True:
                    if stop_flag_ref[0]:
                        return False
                    pause_event.wait()  # blocks here while paused
                    data = resp.read(CHUNK)
                    if not data:
                        break
                    f.write(data)
                    _downloaded_now[0] += len(data)
                    elapsed = time.time() - t0 or 0.001
                    _speed_bps[0] = _downloaded_now[0] / elapsed
            return True

        def _chunked_download(url, zip_path, pause_event, stop_flag_ref):
            """Fetch file size, split into N ranges, parallel download, merge."""
            import time, concurrent.futures
            # HEAD request to get file size
            req = urllib.request.Request(url, method="HEAD")
            try:
                with urllib.request.urlopen(req) as r:
                    total_size = int(r.headers.get("Content-Length", 0))
            except Exception:
                total_size = 0

            _total_now[0] = total_size
            _downloaded_now[0] = 0

            if total_size == 0 or total_size < NUM_CONNECTIONS * 1024 * 1024:
                # Server doesn't support ranges OR file too small — plain download
                CHUNK = 512 * 1024
                req2 = urllib.request.Request(url)
                with urllib.request.urlopen(req2) as resp, open(zip_path, "wb") as f:
                    t0 = time.time()
                    while True:
                        if stop_flag_ref[0]: return False
                        pause_event.wait()
                        data = resp.read(CHUNK)
                        if not data: break
                        f.write(data)
                        _downloaded_now[0] += len(data)
                        elapsed = time.time() - t0 or 0.001
                        _speed_bps[0] = _downloaded_now[0] / elapsed
                return True

            # Split into NUM_CONNECTIONS chunks
            chunk_size = total_size // NUM_CONNECTIONS
            ranges = []
            for ci in range(NUM_CONNECTIONS):
                b_start = ci * chunk_size
                b_end   = (b_start + chunk_size - 1) if ci < NUM_CONNECTIONS - 1 else total_size - 1
                buf = zip_path.parent / f"{zip_path.name}.part{ci}"
                ranges.append((b_start, b_end, buf))

            stop_flag_ref_list = [False]
            ok = True
            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CONNECTIONS) as ex:
                futs = {
                    ex.submit(_download_chunk, url, zip_path, b_start, b_end, buf, pause_event, stop_flag_ref): (ci, buf)
                    for ci, (b_start, b_end, buf) in enumerate(ranges)
                }
                for fut in concurrent.futures.as_completed(futs):
                    if not fut.result():
                        ok = False
                        for f2 in futs: f2.cancel()
                        break

            if not ok or stop_flag_ref[0]:
                for _, _, buf in ranges:
                    try: buf.unlink()
                    except: pass
                return False

            # Merge parts in order
            with open(zip_path, "wb") as out:
                for _, _, buf in sorted(ranges, key=lambda x: x[2]):
                    with open(buf, "rb") as inp:
                        shutil.copyfileobj(inp, out)
                    buf.unlink()
            return True

        # ── PRODUCER: download parts sequentially, queue for extraction ──
        import queue as _queue
        _extract_queue  = _queue.Queue()
        _SENTINEL       = None

        dl_progress_val = 0.0
        dl_status_text  = ""
        ex_progress_val = 0.0
        ex_status_text  = ""
        dl_done         = False
        ex_done         = False
        dl_error        = None
        ex_error        = None

        stop_flag_ref = [False]
        def download_thread():
            nonlocal dl_progress_val, dl_status_text, dl_done, dl_error
            import time
            try:
                completed_parts = set()
                if completed_file.exists():
                    with open(completed_file, "r") as f:
                        completed_parts = {l.strip() for l in f if l.strip()}

                parts_to_download = [
                    i for i in range(start_part, end_part + 1)
                    if f"Kinetics700_part_{i:03d}.zip" not in completed_parts
                ]
                total_parts_count = len(parts_to_download)

                if total_parts_count == 0:
                    dl_status_text = "All selected parts are already downloaded!"
                    dl_done = True
                    _extract_queue.put(_SENTINEL)
                    return

                for idx, i in enumerate(parts_to_download):
                    if self._kin_stop_flag:
                        dl_error = "STOPPED"
                        _extract_queue.put(_SENTINEL)
                        return

                    part_name = f"Kinetics700_part_{i:03d}.zip"
                    url       = f"https://huggingface.co/datasets/atalaydenknalbant/Kinetics-700/resolve/main/{part_name}"
                    zip_path  = temp_dir / part_name

                    stop_flag_ref[0] = False
                    _downloaded_now[0] = 0
                    _total_now[0] = 0
                    _speed_bps[0] = 0.0

                    # Live status updater mini-thread
                    _dl_part_done = [False]
                    def _status_updater(idx=idx, i=i, _done=_dl_part_done):
                        while not _done[0] and not self._kin_stop_flag:
                            self._kin_pause_event.wait()
                            dn  = _downloaded_now[0]
                            tot = _total_now[0]
                            spd = _speed_bps[0]
                            if tot > 0:
                                pct     = min(dn / tot, 1.0)
                                overall = (idx + pct) / total_parts_count
                                speed_str  = f"{spd/1024**2:.1f} MB/s" if spd > 0 else "..."
                                paused_str = " [PAUSED]" if not self._kin_pause_event.is_set() else ""
                                _set_dl_status(
                                    overall,
                                    f"↓ Part {i} ({idx+1}/{total_parts_count}){paused_str}: "
                                    f"{dn/1024**3:.2f}/{tot/1024**3:.2f} GB "
                                    f"({pct*100:.1f}%) — {speed_str} — {NUM_CONNECTIONS}x"
                                )
                            time.sleep(0.3)

                    def _set_dl_status(prog, txt):
                        nonlocal dl_progress_val, dl_status_text
                        dl_progress_val = prog
                        dl_status_text  = txt

                    threading.Thread(target=_status_updater, daemon=True).start()

                    ok = _chunked_download(url, zip_path, self._kin_pause_event, stop_flag_ref)
                    _dl_part_done[0] = True

                    if self._kin_stop_flag or not ok:
                        dl_error = "STOPPED"
                        _extract_queue.put(_SENTINEL)
                        return

                    # Hand off to extraction pipeline (non-blocking)
                    dl_status_text = f"↓ Part {i} downloaded — queuing for extraction..."
                    _extract_queue.put((i, part_name, zip_path, idx, total_parts_count))

                dl_status_text = f"✅ All {total_parts_count} parts downloaded!"
                dl_done = True
                _extract_queue.put(_SENTINEL)   # signal consumer to finish

            except Exception as e:
                dl_error = str(e)
                dl_done = True
                _extract_queue.put(_SENTINEL)
            finally:
                stop_flag_ref[0] = True

        # ── CONSUMER: extract parts as they arrive in the queue ──────────
        def extract_thread():
            nonlocal ex_progress_val, ex_status_text, ex_done, ex_error
            extracted_count = 0
            try:
                while True:
                    item = _extract_queue.get()
                    if item is _SENTINEL:
                        break
                    i, part_name, zip_path, idx, total_parts_count = item

                    ex_status_text = f"⚙ Extracting Part {i} ({extracted_count+1}/{total_parts_count})..."
                    ex_progress_val = extracted_count / max(total_parts_count, 1)

                    result = subprocess.run([
                        "C:\\Program Files\\7-Zip\\7z.exe",
                        "x", str(zip_path), f"-o{p}", "-y"
                    ], capture_output=True, text=True)

                    if result.returncode != 0:
                        raise Exception(f"7z failed for {part_name}: {result.stderr}")

                    if zip_path.exists():
                        zip_path.unlink()

                    with open(completed_file, "a") as f:
                        f.write(f"{part_name}\n")

                    extracted_count += 1
                    ex_progress_val = extracted_count / max(total_parts_count, 1)
                    ex_status_text  = f"⚙ Extracted Part {i} ✓  ({extracted_count}/{total_parts_count} done)"

                ex_status_text = f"✅ All {extracted_count} parts extracted!"
            except Exception as e:
                ex_error = str(e)
            finally:
                ex_done = True

        def check_ui():
            all_done  = dl_done and ex_done
            any_error = dl_error or ex_error

            # Update download row
            self.kin_dl_progress.set(dl_progress_val)
            self.kin_dl_lbl.configure(text=dl_status_text)

            # Update extraction row — show row itself the first time
            if ex_status_text:
                if not self.kin_ex_row.winfo_ismapped():
                    self.kin_ex_row.pack(fill="x", pady=(2, 0))
                self.kin_ex_progress.pack(side="left", padx=(8, 0))
                self.kin_ex_lbl.pack(side="left", padx=(5, 0))
                self.kin_ex_progress.set(ex_progress_val)
                self.kin_ex_lbl.configure(text=ex_status_text)

            if dl_error == "STOPPED":
                self._finish_kin_ui(stopped=True)
            elif any_error:
                err = dl_error or ex_error
                self._finish_kin_ui(error=err)
            elif all_done:
                self._finish_kin_ui()
            else:
                self.after(200, check_ui)

        threading.Thread(target=download_thread, daemon=True).start()
        threading.Thread(target=extract_thread,  daemon=True).start()
        check_ui()

    def _finish_kin_ui(self, stopped=False, error=None):
        self._kin_downloading = False
        self.kin_dl_btn.configure(state="normal")
        try:
            self.kin_ctrl_row.pack_forget()
        except: pass
        self.kin_pause_btn.grid_forget()
        self.kin_stop_btn.grid_forget()
        self.kin_dl_progress.pack_forget()
        self.kin_ex_row.pack_forget()
        self.kin_ex_progress.pack_forget()
        self.kin_ex_lbl.pack_forget()
        if stopped:
            self.kin_dl_lbl.configure(
                text="Download stopped — completed parts saved, restart to resume.",
                text_color=COLORS["warning"])
        elif error:
            self.kin_dl_lbl.configure(text=f"Error: {error}", text_color=COLORS["error"])
        else:
            self.kin_dl_lbl.configure(
                text="✅ All parts downloaded & extracted!",
                text_color=COLORS["success"])
            if hasattr(self, "kin_load_classes"):
                self.kin_load_classes()

    def _toggle_kin_pause(self):
        if not self._kin_downloading:
            return
        if self._kin_pause_event.is_set():
            # Pause: clear the event so all chunk threads block at .wait()
            self._kin_pause_event.clear()
            self.kin_pause_btn.configure(text="▶ Resume")
        else:
            # Resume: set the event so threads unblock
            self._kin_pause_event.set()
            self.kin_pause_btn.configure(text="⏸ Pause")

    def _stop_kin_download(self):
        if not self._kin_downloading:
            return
        self._kin_stop_flag = True
        self._kin_pause_event.set()  # unblock any paused threads so they can see the stop flag

    def _extract_only_kinetics(self, dl_root_str, target_dir_str, stats_lbl):
        if self._kin_downloading:
            return
            
        p = Path(target_dir_str)
        dl_root = Path(dl_root_str)
        temp_zip = dl_root / "temp_zip"
        
        if not temp_zip.exists() or not p.exists():
            return
            
        orphans = list(temp_zip.glob("Kinetics700_part_*.zip"))
        if not orphans:
            return
            
        # Prepare UI
        self._kin_downloading = True
        self.kin_dl_btn.configure(state="disabled")
        self.kin_extract_btn.configure(state="disabled")
        try:
            self.kin_ctrl_row.pack_forget()
        except: pass
        self.kin_dl_progress.pack(fill="x", padx=12, pady=(6,0))
        self.kin_dl_lbl.pack(fill="x", padx=12)
        self.kin_dl_progress.set(0)
        self.kin_dl_lbl.configure(text=f"Preparing to extract {len(orphans)} parts...", text_color=COLORS["text"])
        
        completed_file = dl_root / "completed_parts.txt"
        completed_file.touch(exist_ok=True)
        
        ex_progress_val = 0.0
        ex_status_text = ""
        ex_done = False
        ex_error = None
        
        def extract_thread():
            nonlocal ex_progress_val, ex_status_text, ex_done, ex_error
            try:
                total_orphans = len(orphans)
                for idx, zip_path in enumerate(orphans):
                    part_name = zip_path.name
                    
                    ex_progress_val = idx / total_orphans
                    ex_status_text = f"Extracting {part_name} ({idx+1}/{total_orphans})..."
                    
                    result = subprocess.run([
                        "C:\\Program Files\\7-Zip\\7z.exe", 
                        "x", str(zip_path), f"-o{p}", "-y"
                    ], capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        raise Exception(f"7z extraction failed for {part_name}: {result.stderr}")
                        
                    # Delete zip
                    ex_status_text = f"Cleaning up {part_name}..."
                    if zip_path.exists():
                        zip_path.unlink()
                        
                    # Save state
                    with open(completed_file, "a") as f:
                        f.write(f"{part_name}\n")
                        
                ex_progress_val = 1.0
                ex_status_text = f"Successfully extracted and cleaned up {total_orphans} parts!"
            except Exception as e:
                ex_error = str(e)
            finally:
                ex_done = True
                
        def check_ui():
            if ex_error:
                self._kin_downloading = False
                self.kin_dl_btn.configure(state="normal")
                self.kin_extract_btn.configure(state="normal")
                self.kin_dl_lbl.configure(text=f"Error: {ex_error}", text_color=COLORS["error"])
                self.kin_dl_progress.pack_forget()
            elif ex_done:
                self._kin_downloading = False
                self.kin_dl_btn.configure(state="normal")
                self.kin_extract_btn.configure(state="normal")
                self.kin_extract_btn.pack_forget() # Hide it since orphans are gone
                self.kin_dl_lbl.configure(text=ex_status_text, text_color=COLORS["success"])
                self.kin_dl_progress.pack_forget()
                
                if hasattr(self, "kin_load_classes"):
                    self.kin_load_classes()
            else:
                self.kin_dl_progress.set(ex_progress_val)
                self.kin_dl_lbl.configure(text=ex_status_text)
                self.after(200, check_ui)
                
        threading.Thread(target=extract_thread, daemon=True).start()
        check_ui()
