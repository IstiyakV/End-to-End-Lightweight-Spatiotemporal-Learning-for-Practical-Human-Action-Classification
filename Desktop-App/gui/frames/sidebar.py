"""Sidebar — premium navigation with Unicode icons + active glow."""

import customtkinter as ctk
from gui.theme import COLORS
from gui.settings import load_settings, save_setting

NAV_ITEMS = [
    ("dashboard",  "\u2302",  "Dashboard"),      # ⌂
    ("datasets",   "\u2630",  "Datasets"),        # ☰
    ("network",    "\u2B21",  "Network"),         # ⬡
    ("transfer",   "\u21C6",  "Transfer Learning"), # ⇄
    ("training",   "\u2699",  "Training"),        # ⚙
    ("retrain",    "\u21BA",  "Retrain Model"),   # ↺
    ("monitor",    "\u25B6",  "Monitor"),         # ▶
    ("tester",     "\u2316",  "Test Model"),      # ⌖
    ("benchmark",  "\u2261",  "SOTA Bench"),      # ≡
    ("results",    "\u2637",  "Results"),         # ☷
]

class Sidebar(ctk.CTkFrame):
    def __init__(self, parent, switch_callback):
        super().__init__(parent, width=220, fg_color=COLORS["sidebar"], corner_radius=0)
        self.switch_callback = switch_callback
        self.buttons = {}
        self.indicators = {}
        self.grid_propagate(False)

        # ── Logo area ──
        logo_frame = ctk.CTkFrame(self, fg_color="transparent", height=80)
        logo_frame.pack(fill="x", pady=(20, 0))
        logo_frame.pack_propagate(False)

        # Icon circle
        icon_circle = ctk.CTkFrame(logo_frame, width=42, height=42,
                                    fg_color=COLORS["accent"], corner_radius=21)
        icon_circle.place(relx=0.5, rely=0.35, anchor="center")
        ctk.CTkLabel(icon_circle, text="H", font=("Segoe UI", 18, "bold"),
                     text_color=("#ffffff", "#ffffff")).place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(logo_frame, text="HAR Control Center",
                     font=("Segoe UI", 13, "bold"), text_color=COLORS["text"]
                     ).place(relx=0.5, rely=0.78, anchor="center")

        # Divider
        ctk.CTkFrame(self, height=1, fg_color=COLORS["border"]).pack(fill="x", padx=18, pady=(12, 16))

        # ── Section label ──
        ctk.CTkLabel(self, text="NAVIGATION", font=("Segoe UI", 9),
                     text_color=COLORS["text_dim"], anchor="w"
                     ).pack(anchor="w", padx=24, pady=(0, 8))

        # ── Nav buttons ──
        for key, icon, label in NAV_ITEMS:
            container = ctk.CTkFrame(self, fg_color="transparent", height=44)
            container.pack(fill="x", padx=12, pady=2)
            container.pack_propagate(False)

            # Active indicator bar (left edge)
            indicator = ctk.CTkFrame(container, width=3, fg_color="transparent",
                                      corner_radius=2)
            indicator.pack(side="left", fill="y", padx=(0, 0))
            self.indicators[key] = indicator

            btn = ctk.CTkButton(
                container,
                text=f"  {icon}   {label}",
                anchor="w",
                font=("Segoe UI", 13),
                height=40,
                fg_color="transparent",
                text_color=COLORS["text_dim"],
                hover_color=COLORS["sidebar_hover"],
                corner_radius=10,
                command=lambda k=key: self.switch_callback(k),
            )
            btn.pack(side="left", fill="both", expand=True, padx=(4, 8))
            self.buttons[key] = btn

        # ── Bottom section ──
        spacer = ctk.CTkFrame(self, fg_color="transparent")
        spacer.pack(fill="both", expand=True)
        
        # Theme Toggle
        settings = load_settings()
        current_mode = settings.get("appearance_mode", "dark")
        
        def toggle_theme(mode):
            self.master.change_theme(mode)
            
        theme_btn = ctk.CTkSegmentedButton(self, values=["☀️ Light", "🌙 Dark"], 
                                           command=toggle_theme,
                                           font=("Segoe UI", 11, "bold"),
                                           height=34,
                                           fg_color=COLORS["input_bg"],
                                           selected_color=COLORS["accent"],
                                           selected_hover_color=COLORS["accent_hover"],
                                           text_color=COLORS["text_dim"])
        theme_btn.pack(fill="x", padx=18, pady=(0, 15))
        theme_btn.set("🌙 Dark" if current_mode == "dark" else "☀️ Light")

        # Divider
        ctk.CTkFrame(self, height=1, fg_color=COLORS["border"]).pack(fill="x", padx=18, pady=(0, 10))

        # GPU status mini-card
        gpu_card = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=10, height=55)
        gpu_card.pack(fill="x", padx=14, pady=(0, 8))
        gpu_card.pack_propagate(False)

        try:
            from gui.services.gpu_service import detect_gpus
            gpus = detect_gpus()
            if gpus:
                g = gpus[0]
                gpu_name = g["name"].replace("NVIDIA GeForce ", "")
                ctk.CTkLabel(gpu_card, text=f"\u2622  {gpu_name}",
                             font=("Segoe UI", 11, "bold"), text_color=COLORS["text"]
                             ).pack(anchor="w", padx=12, pady=(8, 0))
                ctk.CTkLabel(gpu_card, text=f"{g['vram_mb']} MB VRAM",
                             font=("Segoe UI", 9), text_color=COLORS["text_dim"]
                             ).pack(anchor="w", padx=12)
            else:
                ctk.CTkLabel(gpu_card, text="CPU Mode",
                             font=("Segoe UI", 11), text_color=COLORS["text_dim"]
                             ).pack(anchor="w", padx=12, pady=12)
        except:
            pass

        # Version
        ctk.CTkLabel(self, text="v1.0.0", font=("Segoe UI", 9),
                     text_color=COLORS["text_dim"]).pack(pady=(4, 12))

    def set_active(self, name: str):
        for key, btn in self.buttons.items():
            if key == name:
                btn.configure(fg_color=COLORS["accent"], text_color=("#ffffff", "#ffffff"), hover_color=COLORS["accent_hover"])
                self.indicators[key].configure(fg_color=COLORS["accent"])
            else:
                btn.configure(fg_color="transparent", text_color=COLORS["text_dim"], hover_color=COLORS["sidebar_hover"])
                self.indicators[key].configure(fg_color="transparent")
