"""Network Architect — visual neural network builder (v2).

Redesigned for user-friendliness:
  • One-click architecture presets (Paper Default, Lightweight, Deep, Plain)
  • Vertical flow diagram that looks like a real neural network pipeline
  • Live model summary table with shapes and param counts
  • Enhanced properties panel with tooltips, ranges, and validation
  • Insert-between "+" buttons for intuitive layer insertion
"""

import customtkinter as ctk
import tkinter as tk
from pathlib import Path
import sys
import math

from gui.theme import COLORS

# ---------------------------------------------------------------------------
# Backend — graceful fallback if model_builder is not yet available
# ---------------------------------------------------------------------------
_BUILDER_AVAILABLE = False
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from har.model_builder import (
        LAYER_TYPES,
        LAYER_COLORS,
        PRESET_META,
        LayerConfig,
        ArchitectureConfig,
        default_architecture,
        preset_architectures,
        validate_architecture,
        get_default_params,
        compute_shapes,
        build_from_config,
        count_params,
        list_saved_architectures,
        ARCH_DIR,
    )
    _BUILDER_AVAILABLE = True
except Exception:
    LAYER_TYPES = {}
    LAYER_COLORS = {}
    PRESET_META = {}
    ARCH_DIR = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(color_tuple):
    """Return the single hex colour for the current appearance mode."""
    if isinstance(color_tuple, (list, tuple)):
        mode = ctk.get_appearance_mode()
        return color_tuple[1] if mode == "Dark" else color_tuple[0]
    return color_tuple


def _lighten(hex_color: str, factor: float = 0.35) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _darken(hex_color: str, factor: float = 0.25) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:], 16)
    r = int(r * (1 - factor))
    g = int(g * (1 - factor))
    b = int(b * (1 - factor))
    return f"#{r:02x}{g:02x}{b:02x}"


# Parameter tooltips — explains what each setting does for non-experts
_TOOLTIPS = {
    "filters": "Number of feature maps the layer learns.\nMore filters → more capacity but slower training.",
    "kernel_size": "Size of the 3D convolution window.\n3 is the most common choice; 5 and 7 see more context.",
    "stride": "Step size of the convolution.\n1 = preserve dimensions, 2 = halve dimensions.",
    "pool_size": "Factor to reduce each dimension.\n2×2×2 halves temporal, height, and width.",
    "rate": "Fraction of neurons randomly turned off during training.\nPrevents overfitting. 0.2–0.5 is typical.",
    "units": "Number of output neurons.\nOverridden by the number of classes during training.",
}

_RANGES = {
    "filters": "Typical: 16–256 (multiples of 8)",
    "kernel_size": "3 is standard. Use 5 or 7 for larger receptive fields.",
    "rate": "Recommended: 0.2–0.5",
    "units": "Auto-set to class count at training time",
}


# ---------------------------------------------------------------------------
# Main frame
# ---------------------------------------------------------------------------

class NetworkArchitectFrame(ctk.CTkFrame):
    """Visual neural-network architecture editor (v2)."""

    # Canvas layout constants — vertical single-column flow
    BLOCK_W = 210
    BLOCK_H = 60
    PAD_Y = 14          # gap between blocks (arrow zone)
    INSERT_H = 18       # height of "+" insert button zone
    MARGIN_TOP = 24
    MARGIN_LEFT = 30

    def __init__(self, parent, app):
        super().__init__(parent, fg_color=COLORS["bg"], corner_radius=0)
        self.app = app
        self.architecture = None
        self.selected_idx = None
        self._block_ids: list[list[int]] = []
        self._shapes: list = []
        self._warnings: list[str] = []

        if not _BUILDER_AVAILABLE:
            self._build_fallback()
            return

        self._build_header()
        self._build_presets_bar()
        self._build_toolbar()
        self._build_main_area()
        self._build_status_bar()

        # Load the default architecture on startup
        self._load_architecture(default_architecture())

    # ------------------------------------------------------------------
    # Fallback when backend isn't available
    # ------------------------------------------------------------------
    def _build_fallback(self):
        holder = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=14,
                              border_width=1, border_color=COLORS["card_border"])
        holder.pack(expand=True, padx=60, pady=60)
        ctk.CTkLabel(holder, text="⚠  Network Architect Unavailable",
                     font=("Segoe UI", 20, "bold"),
                     text_color=COLORS["warning"]).pack(padx=40, pady=(30, 8))
        ctk.CTkLabel(holder, text="The model_builder backend module could not be loaded.\n"
                                  "Please ensure  har/model_builder.py  exists and is importable.",
                     font=("Segoe UI", 12), text_color=COLORS["text_dim"],
                     justify="center").pack(padx=40, pady=(0, 30))

    # ------------------------------------------------------------------
    # 1. Header
    # ------------------------------------------------------------------
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=30, pady=(25, 0))
        ctk.CTkLabel(header, text="Network Architect",
                     font=("Segoe UI", 26, "bold"),
                     text_color=COLORS["text"]).pack(anchor="w")
        ctk.CTkLabel(header, text="Design, visualise and export custom CNN architectures",
                     font=("Segoe UI", 12),
                     text_color=COLORS["text_dim"]).pack(anchor="w")

    # ------------------------------------------------------------------
    # 2. Presets bar — one-click architecture templates
    # ------------------------------------------------------------------
    def _build_presets_bar(self):
        bar = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=14,
                           border_width=1, border_color=COLORS["card_border"])
        bar.pack(fill="x", padx=20, pady=(12, 0))

        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.pack(fill="x", padx=12, pady=10)

        ctk.CTkLabel(inner, text="⚡ Quick Start",
                     font=("Segoe UI", 12, "bold"),
                     text_color=COLORS["text"]).pack(side="left", padx=(4, 12))

        presets = preset_architectures()
        for name, arch in presets.items():
            meta = PRESET_META.get(name, {})
            badge = meta.get("badge", "")
            badge_color = meta.get("badge_color", _resolve(COLORS["accent"]))
            desc = meta.get("description", "")

            # Compute param count for the preset
            try:
                m = build_from_config(arch, num_classes=101)
                total, _ = count_params(m)
                param_text = f"{total:,} params"
            except Exception:
                param_text = ""

            # Card frame for each preset
            card = ctk.CTkFrame(inner, fg_color=_resolve(COLORS["input_bg"]),
                                corner_radius=10, border_width=1,
                                border_color=_resolve(COLORS["border"]))
            card.pack(side="left", padx=4, pady=2)

            card_inner = ctk.CTkFrame(card, fg_color="transparent")
            card_inner.pack(padx=10, pady=8)

            # Top row: name + badge
            top = ctk.CTkFrame(card_inner, fg_color="transparent")
            top.pack(fill="x")
            ctk.CTkLabel(top, text=name, font=("Segoe UI", 11, "bold"),
                         text_color=_resolve(COLORS["text"])).pack(side="left")
            ctk.CTkLabel(top, text=f"  {badge}",
                         font=("Segoe UI", 9, "bold"),
                         text_color=badge_color).pack(side="left")

            # Description + params
            ctk.CTkLabel(card_inner, text=desc,
                         font=("Segoe UI", 9),
                         text_color=_resolve(COLORS["text_dim"])).pack(anchor="w")
            if param_text:
                ctk.CTkLabel(card_inner, text=param_text,
                             font=("Segoe UI", 9),
                             text_color=_resolve(COLORS["text_dim"])).pack(anchor="w")

            # Use button
            ctk.CTkButton(card_inner, text="Use This",
                          font=("Segoe UI", 10, "bold"),
                          fg_color=badge_color,
                          hover_color=_lighten(badge_color, 0.15),
                          text_color="#ffffff", height=26, width=90,
                          corner_radius=6,
                          command=lambda a=arch: self._load_architecture(
                              ArchitectureConfig.from_dict(a.to_dict())  # deep copy
                          )).pack(pady=(4, 0))

    # ------------------------------------------------------------------
    # 3. Toolbar
    # ------------------------------------------------------------------
    def _build_toolbar(self):
        bar = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=14,
                           border_width=1, border_color=COLORS["card_border"])
        bar.pack(fill="x", padx=20, pady=(8, 0))

        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.pack(fill="x", padx=12, pady=8)

        # -- LEFT: add-layer buttons --
        left = ctk.CTkFrame(inner, fg_color="transparent")
        left.pack(side="left")

        ctk.CTkLabel(left, text="Add Layer:",
                     font=("Segoe UI", 11, "bold"),
                     text_color=_resolve(COLORS["text_dim"])).pack(side="left", padx=(0, 8))

        add_types = [
            ("conv3d_r21d", "+ R(2+1)D"),
            ("conv3d_plain", "+ Conv3D"),
            ("maxpool3d", "+ Pool"),
            ("batchnorm3d", "+ BN"),
            ("dropout", "+ Drop"),
        ]
        for ltype, label in add_types:
            color = LAYER_COLORS.get(ltype, _resolve(COLORS["accent"]))
            ctk.CTkButton(
                left, text=label, font=("Segoe UI", 10, "bold"),
                fg_color=color, hover_color=_lighten(color, 0.18),
                text_color="#ffffff", height=28, width=len(label) * 9 + 14,
                corner_radius=8,
                command=lambda t=ltype: self._add_layer(t),
            ).pack(side="left", padx=2)

        # -- RIGHT: actions --
        right = ctk.CTkFrame(inner, fg_color="transparent")
        right.pack(side="right")

        action_btns = [
            ("▲ Up",    lambda: self._move_layer(-1)),
            ("▼ Down",  lambda: self._move_layer(1)),
            ("🗑 Del",   self._delete_layer),
            ("↺ Reset", self._reset_default),
        ]
        for label, cmd in action_btns:
            ctk.CTkButton(
                right, text=label, font=("Segoe UI", 10),
                fg_color=_resolve(COLORS["input_bg"]),
                hover_color=_resolve(COLORS["sidebar_hover"]),
                text_color=_resolve(COLORS["text"]),
                height=28, width=66, corner_radius=8, command=cmd,
            ).pack(side="left", padx=2)

        ctk.CTkFrame(right, width=10, fg_color="transparent").pack(side="left")

        # Save / Load
        ctk.CTkButton(
            right, text="💾 Save", font=("Segoe UI", 10, "bold"),
            fg_color=_resolve(COLORS["success"]),
            hover_color=_lighten(_resolve(COLORS["success"]), 0.15),
            text_color="#ffffff", height=28, width=72, corner_radius=8,
            command=self._save_architecture_dialog,
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            right, text="📂 Load", font=("Segoe UI", 10, "bold"),
            fg_color=_resolve(COLORS["accent"]),
            hover_color=_resolve(COLORS["accent_hover"]),
            text_color="#ffffff", height=28, width=72, corner_radius=8,
            command=self._load_architecture_dialog,
        ).pack(side="left", padx=2)

    # ------------------------------------------------------------------
    # 4. Main area (canvas + properties + summary)
    # ------------------------------------------------------------------
    def _build_main_area(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=(8, 0))
        main.grid_columnconfigure(0, weight=5)
        main.grid_columnconfigure(1, weight=4)
        main.grid_rowconfigure(0, weight=1)

        # --- LEFT: canvas ---
        canvas_card = ctk.CTkFrame(main, fg_color=COLORS["card"], corner_radius=14,
                                   border_width=1, border_color=COLORS["card_border"])
        canvas_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)

        ctk.CTkLabel(canvas_card, text="  Architecture Flow",
                     font=("Segoe UI", 12, "bold"),
                     text_color=COLORS["text_dim"], anchor="w").pack(fill="x", padx=14, pady=(10, 4))

        self.canvas = tk.Canvas(
            canvas_card,
            bg=_resolve(COLORS["console_bg"]),
            highlightthickness=0,
            relief="flat",
        )
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        v_scroll = ctk.CTkScrollbar(canvas_card, command=self.canvas.yview,
                                    button_color=_resolve(COLORS["accent"]),
                                    button_hover_color=_resolve(COLORS["accent_hover"]))
        v_scroll.place(relx=1.0, rely=0.08, relheight=0.88, anchor="ne", x=-4)
        self.canvas.configure(yscrollcommand=v_scroll.set)

        self.canvas.bind("<Button-1>", self._on_canvas_bg_click)
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-e.delta / 120), "units"))

        # --- RIGHT: properties + summary stacked vertically ---
        right_frame = ctk.CTkFrame(main, fg_color="transparent")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=0)
        right_frame.grid_rowconfigure(0, weight=3)
        right_frame.grid_rowconfigure(1, weight=2)
        right_frame.grid_columnconfigure(0, weight=1)

        # Properties panel (top-right)
        self.props_card = ctk.CTkFrame(right_frame, fg_color=COLORS["card"], corner_radius=14,
                                       border_width=1, border_color=COLORS["card_border"])
        self.props_card.grid(row=0, column=0, sticky="nsew", pady=(0, 6))

        self.props_title = ctk.CTkLabel(self.props_card, text="Layer Properties",
                                        font=("Segoe UI", 14, "bold"),
                                        text_color=COLORS["text"])
        self.props_title.pack(anchor="w", padx=16, pady=(14, 4))

        ctk.CTkFrame(self.props_card, height=1,
                     fg_color=COLORS["border"]).pack(fill="x", padx=14, pady=(0, 6))

        self.props_scroll = ctk.CTkScrollableFrame(self.props_card, fg_color="transparent",
                                                    scrollbar_button_color=_resolve(COLORS["border"]),
                                                    scrollbar_button_hover_color=_resolve(COLORS["card_border"]))
        self.props_scroll.pack(fill="both", expand=True, padx=6, pady=(0, 10))

        # Container that gets rebuilt on each selection change
        self.props_container = ctk.CTkFrame(self.props_scroll, fg_color="transparent")
        self.props_container.pack(fill="both", expand=True)

        # Summary table (bottom-right)
        self.summary_card = ctk.CTkFrame(right_frame, fg_color=COLORS["card"], corner_radius=14,
                                          border_width=1, border_color=COLORS["card_border"])
        self.summary_card.grid(row=1, column=0, sticky="nsew", pady=(6, 0))

        ctk.CTkLabel(self.summary_card, text="  Model Summary",
                     font=("Segoe UI", 12, "bold"),
                     text_color=COLORS["text_dim"], anchor="w").pack(fill="x", padx=14, pady=(10, 4))

        self.summary_scroll = ctk.CTkScrollableFrame(
            self.summary_card, fg_color="transparent",
            scrollbar_button_color=_resolve(COLORS["border"]),
            scrollbar_button_hover_color=_resolve(COLORS["card_border"]))
        self.summary_scroll.pack(fill="both", expand=True, padx=6, pady=(0, 10))

        self.summary_container = ctk.CTkFrame(self.summary_scroll, fg_color="transparent")
        self.summary_container.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # 5. Status bar
    # ------------------------------------------------------------------
    def _build_status_bar(self):
        bar = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=14,
                           border_width=1, border_color=COLORS["card_border"], height=36)
        bar.pack(fill="x", padx=20, pady=(6, 12))
        bar.pack_propagate(False)

        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16)

        self.status_params = ctk.CTkLabel(inner, text="Total Parameters: –",
                                          font=("Segoe UI", 10),
                                          text_color=COLORS["text_dim"])
        self.status_params.pack(side="left")

        self.status_warnings = ctk.CTkLabel(inner, text="",
                                             font=("Segoe UI", 10, "bold"),
                                             text_color=COLORS["error"])
        self.status_warnings.pack(side="left", padx=(16, 0))

        self.status_layers = ctk.CTkLabel(inner, text="Layers: –",
                                          font=("Segoe UI", 10),
                                          text_color=COLORS["text_dim"])
        self.status_layers.pack(side="right")

        self.status_name = ctk.CTkLabel(inner, text="",
                                        font=("Segoe UI", 10, "bold"),
                                        text_color=COLORS["accent"])
        self.status_name.pack(expand=True)

    # ==================================================================
    #  CORE METHODS
    # ==================================================================

    def _load_architecture(self, arch):
        """Load an architecture and redraw everything."""
        self.architecture = arch
        self.selected_idx = None
        self._redraw_canvas()
        self._update_properties()
        self._update_summary_table()
        self._update_status()

    # ------------------------------------------------------------------
    #  Canvas drawing — vertical flow
    # ------------------------------------------------------------------

    def _block_position(self, idx: int):
        """Return (cx, cy) centre for block *idx* in a single-column vertical flow."""
        x = self.MARGIN_LEFT + self.BLOCK_W // 2
        # Each block slot = BLOCK_H + PAD_Y (arrow) + INSERT_H (+ button)
        slot_h = self.BLOCK_H + self.PAD_Y + self.INSERT_H
        y = self.MARGIN_TOP + idx * slot_h + self.BLOCK_H // 2
        return x, y

    def _redraw_canvas(self):
        """Clear and redraw all blocks and arrows on the canvas."""
        self.canvas.delete("all")
        self._block_ids.clear()

        if self.architecture is None:
            return

        # Compute output shapes
        try:
            self._shapes = compute_shapes(self.architecture)
        except Exception:
            self._shapes = [None] * (len(self.architecture.layers) + 1)

        # Validate
        try:
            self._warnings = validate_architecture(self.architecture)
        except Exception:
            self._warnings = []

        all_layers = self.architecture.layers
        total = len(all_layers) + 1  # +1 for input

        for i in range(total):
            cx, cy = self._block_position(i)
            is_input = (i == 0)
            is_selected = (i == self.selected_idx)

            if is_input:
                fill = "#4b5563"
                label1 = "INPUT"
                inp = all_layers[0].params if all_layers else {}
                c = inp.get("channels", 3)
                label2 = f"{c} ch × 10 frames × 224×224"
                label3 = ""
            else:
                layer = all_layers[i - 1]
                fill = layer.color
                label1 = layer.display_name
                label2 = layer.summary
                shape = self._shapes[i] if i < len(self._shapes) else None
                label3 = "→ " + "×".join(str(s) for s in shape) if shape else ""

            # Draw rounded rect block
            x0, y0 = cx - self.BLOCK_W // 2, cy - self.BLOCK_H // 2
            x1, y1 = cx + self.BLOCK_W // 2, cy + self.BLOCK_H // 2
            r = 12
            border_col = "#ffffff" if is_selected else _darken(fill, 0.10)
            border_w = 2.5 if is_selected else 1

            ids = []
            rect_id = self._draw_rounded_rect(x0, y0, x1, y1, r, fill, border_col, border_w)
            ids.append(rect_id)

            # Selection glow
            if is_selected:
                glow = self._draw_rounded_rect(x0 - 3, y0 - 3, x1 + 3, y1 + 3, r + 2,
                                               "", _lighten(fill, 0.5), 1.5)
                self.canvas.tag_lower(glow, rect_id)
                ids.append(glow)

            # Layer type icon + text
            t1 = self.canvas.create_text(cx, cy - 14, text=label1,
                                         fill="#ffffff", font=("Segoe UI", 10, "bold"))
            t2 = self.canvas.create_text(cx, cy + 4, text=label2,
                                         fill="#e0e0e0", font=("Segoe UI", 9))
            t3 = self.canvas.create_text(cx, cy + 19, text=label3,
                                         fill="#9ca3af", font=("Segoe UI", 8))
            ids.extend([t1, t2, t3])

            self._block_ids.append(ids)

            # Bind clicks
            for item_id in ids:
                self.canvas.tag_bind(item_id, "<Button-1>",
                                     lambda e, idx=i: self._on_block_click(idx))

        # Draw arrows between blocks + insert buttons
        arrow_color = _resolve(COLORS["text_dim"])
        for i in range(total - 1):
            _, fy = self._block_position(i)
            _, ty = self._block_position(i + 1)
            ax = self.MARGIN_LEFT + self.BLOCK_W // 2

            # Arrow: bottom of block i → top of block i+1
            arrow_start = fy + self.BLOCK_H // 2 + 2
            arrow_end = ty - self.BLOCK_H // 2 - 2
            mid_y = (arrow_start + arrow_end) / 2

            self.canvas.create_line(ax, arrow_start, ax, arrow_end,
                                    fill=arrow_color, width=1.5,
                                    arrow=tk.LAST, arrowshape=(8, 10, 4))

            # "+" insert button
            plus_r = 9
            plus_bg = self.canvas.create_oval(
                ax - plus_r, mid_y - plus_r, ax + plus_r, mid_y + plus_r,
                fill=_resolve(COLORS["input_bg"]), outline=_resolve(COLORS["border"]), width=1)
            plus_txt = self.canvas.create_text(
                ax, mid_y, text="+", fill=_resolve(COLORS["accent"]),
                font=("Segoe UI", 12, "bold"))

            # Bind click on "+" to insert after block i
            for pid in (plus_bg, plus_txt):
                self.canvas.tag_bind(pid, "<Button-1>",
                                     lambda e, pos=i: self._show_insert_menu(e, pos))

        # Update scroll region
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=(0, 0, bbox[2] + 40, bbox[3] + 40))

    def _draw_rounded_rect(self, x0, y0, x1, y1, r, fill, outline, width=1):
        points = [
            x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
            x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
            x0, y1, x0, y1 - r, x0, y0 + r, x0, y0, x0 + r, y0,
        ]
        return self.canvas.create_polygon(
            points, smooth=True,
            fill=fill if fill else "",
            outline=outline, width=width,
        )

    # ------------------------------------------------------------------
    #  Insert menu — appears when "+" is clicked between blocks
    # ------------------------------------------------------------------
    def _show_insert_menu(self, event, after_block_idx):
        """Show a popup menu to pick which layer type to insert."""
        menu = tk.Menu(self.canvas, tearoff=0, font=("Segoe UI", 10),
                       bg=_resolve(COLORS["card"]), fg=_resolve(COLORS["text"]),
                       activebackground=_resolve(COLORS["accent"]),
                       activeforeground="#ffffff")

        insert_types = [
            ("conv3d_r21d",  "Conv3D R(2+1)D"),
            ("conv3d_plain", "Conv3D Plain"),
            ("maxpool3d",    "MaxPool3D"),
            ("batchnorm3d",  "BatchNorm3D"),
            ("dropout",      "Dropout"),
        ]
        for ltype, label in insert_types:
            menu.add_command(
                label=f"  {label}",
                command=lambda t=ltype, pos=after_block_idx: self._insert_layer_at(pos, t))

        # Position near the "+" button
        canvas_x = self.canvas.winfo_rootx() + event.x
        canvas_y = self.canvas.winfo_rooty() + event.y
        menu.tk_popup(canvas_x, canvas_y)

    def _insert_layer_at(self, after_block_idx, layer_type):
        """Insert a new layer after the given block index (0=input)."""
        if self.architecture is None:
            return
        new_layer = LayerConfig(layer_type=layer_type, params=get_default_params(layer_type))
        insert_pos = after_block_idx  # layer list is offset by 1 from block idx (input=0)
        self.architecture.layers.insert(insert_pos, new_layer)
        self.selected_idx = insert_pos + 1
        self._redraw_canvas()
        self._update_properties()
        self._update_summary_table()
        self._update_status()

    # ------------------------------------------------------------------
    #  Click handlers
    # ------------------------------------------------------------------

    def _on_canvas_bg_click(self, event):
        if not self.canvas.find_withtag("current"):
            self.selected_idx = None
            self._redraw_canvas()
            self._update_properties()

    def _on_block_click(self, idx: int):
        self.selected_idx = idx
        self._redraw_canvas()
        self._update_properties()

    # ------------------------------------------------------------------
    #  Layer manipulation
    # ------------------------------------------------------------------

    def _add_layer(self, layer_type: str):
        """Add a new layer after the selected position (or before last Dense)."""
        if self.architecture is None:
            return
        new_layer = LayerConfig(layer_type=layer_type, params=get_default_params(layer_type))

        layers = self.architecture.layers
        if self.selected_idx is not None and self.selected_idx > 0:
            insert_pos = self.selected_idx
        else:
            insert_pos = len(layers)
            for i in reversed(range(len(layers))):
                if layers[i].layer_type == "dense":
                    insert_pos = i
                    break

        layers.insert(insert_pos, new_layer)
        self.selected_idx = insert_pos + 1
        self._redraw_canvas()
        self._update_properties()
        self._update_summary_table()
        self._update_status()

    def _delete_layer(self):
        if self.architecture is None or self.selected_idx is None:
            return
        if self.selected_idx == 0:
            return

        layer_idx = self.selected_idx - 1
        layers = self.architecture.layers

        if layer_idx < 0 or layer_idx >= len(layers):
            return

        layer = layers[layer_idx]
        if layer.layer_type == "dense":
            dense_count = sum(1 for l in layers if l.layer_type == "dense")
            if dense_count <= 1:
                return

        layers.pop(layer_idx)
        self.selected_idx = None
        self._redraw_canvas()
        self._update_properties()
        self._update_summary_table()
        self._update_status()

    def _move_layer(self, direction: int):
        if self.architecture is None or self.selected_idx is None:
            return
        if self.selected_idx == 0:
            return

        layer_idx = self.selected_idx - 1
        layers = self.architecture.layers
        new_idx = layer_idx + direction

        if new_idx < 0 or new_idx >= len(layers):
            return

        layers[layer_idx], layers[new_idx] = layers[new_idx], layers[layer_idx]
        self.selected_idx = new_idx + 1
        self._redraw_canvas()
        self._update_properties()
        self._update_summary_table()
        self._update_status()

    def _reset_default(self):
        self._load_architecture(default_architecture())

    # ------------------------------------------------------------------
    #  Save / Load dialogs
    # ------------------------------------------------------------------

    def _save_architecture_dialog(self):
        if self.architecture is None:
            return

        dialog = ctk.CTkToplevel(self)
        dialog.title("Save Architecture")
        dialog.geometry("380x200")
        dialog.resizable(False, False)
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()

        frame = ctk.CTkFrame(dialog, fg_color=COLORS["card"], corner_radius=0)
        frame.pack(fill="both", expand=True)

        ctk.CTkLabel(frame, text="Save Architecture",
                     font=("Segoe UI", 18, "bold"),
                     text_color=COLORS["text"]).pack(padx=24, pady=(24, 8))
        ctk.CTkLabel(frame, text="Enter a name for this architecture:",
                     font=("Segoe UI", 11),
                     text_color=COLORS["text_dim"]).pack(padx=24, pady=(0, 10))

        name_var = ctk.StringVar(value=self.architecture.name or "custom")
        entry = ctk.CTkEntry(frame, textvariable=name_var, font=("Segoe UI", 12),
                             fg_color=COLORS["input_bg"], border_color=COLORS["input_border"],
                             corner_radius=8, width=300)
        entry.pack(padx=24)
        entry.focus_set()

        status = ctk.CTkLabel(frame, text="", font=("Segoe UI", 10),
                              text_color=COLORS["success"])
        status.pack(pady=(4, 0))

        def do_save():
            name = name_var.get().strip()
            if not name:
                status.configure(text="Name cannot be empty.", text_color=COLORS["error"])
                return
            self.architecture.name = name
            try:
                path = Path(ARCH_DIR) / f"{name}.json"
                self.architecture.save(path)
                status.configure(text=f"Saved to {path.name} ✓", text_color=COLORS["success"])
                self._update_status()
                dialog.after(800, dialog.destroy)
            except Exception as exc:
                status.configure(text=f"Error: {exc}", text_color=COLORS["error"])

        btn_row = ctk.CTkFrame(frame, fg_color="transparent")
        btn_row.pack(pady=(10, 16))
        ctk.CTkButton(btn_row, text="Cancel", fg_color=_resolve(COLORS["input_bg"]),
                      hover_color=_resolve(COLORS["sidebar_hover"]),
                      text_color=_resolve(COLORS["text"]),
                      width=100, corner_radius=8,
                      command=dialog.destroy).pack(side="left", padx=6)
        ctk.CTkButton(btn_row, text="💾 Save", fg_color=_resolve(COLORS["success"]),
                      hover_color=_lighten(_resolve(COLORS["success"]), 0.15),
                      text_color="#ffffff", width=100, corner_radius=8,
                      command=do_save).pack(side="left", padx=6)

    def _load_architecture_dialog(self):
        saved = list_saved_architectures()
        if not saved:
            self._show_toast("No saved architectures found.")
            return

        dialog = ctk.CTkToplevel(self)
        dialog.title("Load Architecture")
        dialog.geometry("380x220")
        dialog.resizable(False, False)
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()

        frame = ctk.CTkFrame(dialog, fg_color=COLORS["card"], corner_radius=0)
        frame.pack(fill="both", expand=True)

        ctk.CTkLabel(frame, text="Load Architecture",
                     font=("Segoe UI", 18, "bold"),
                     text_color=COLORS["text"]).pack(padx=24, pady=(24, 8))
        ctk.CTkLabel(frame, text="Select a saved architecture:",
                     font=("Segoe UI", 11),
                     text_color=COLORS["text_dim"]).pack(padx=24, pady=(0, 10))

        choice_var = ctk.StringVar(value=saved[0])
        ctk.CTkOptionMenu(frame, variable=choice_var, values=saved,
                          font=("Segoe UI", 12),
                          fg_color=COLORS["input_bg"],
                          button_color=_resolve(COLORS["accent"]),
                          corner_radius=8, width=300).pack(padx=24)

        status = ctk.CTkLabel(frame, text="", font=("Segoe UI", 10),
                              text_color=COLORS["error"])
        status.pack(pady=(6, 0))

        def do_load():
            name = choice_var.get()
            try:
                path = Path(ARCH_DIR) / f"{name}.json"
                arch = ArchitectureConfig.load(path)
                self._load_architecture(arch)
                dialog.destroy()
            except Exception as exc:
                status.configure(text=f"Error: {exc}")

        btn_row = ctk.CTkFrame(frame, fg_color="transparent")
        btn_row.pack(pady=(12, 16))
        ctk.CTkButton(btn_row, text="Cancel", fg_color=_resolve(COLORS["input_bg"]),
                      hover_color=_resolve(COLORS["sidebar_hover"]),
                      text_color=_resolve(COLORS["text"]),
                      width=100, corner_radius=8,
                      command=dialog.destroy).pack(side="left", padx=6)
        ctk.CTkButton(btn_row, text="📂 Load", fg_color=_resolve(COLORS["accent"]),
                      hover_color=_resolve(COLORS["accent_hover"]),
                      text_color="#ffffff", width=100, corner_radius=8,
                      command=do_load).pack(side="left", padx=6)

    def _show_toast(self, msg: str):
        toast = ctk.CTkToplevel(self)
        toast.overrideredirect(True)
        toast.attributes("-topmost", True)
        x = self.winfo_rootx() + self.winfo_width() // 2 - 140
        y = self.winfo_rooty() + 60
        toast.geometry(f"280x40+{x}+{y}")
        lbl = ctk.CTkLabel(toast, text=msg, font=("Segoe UI", 11),
                           fg_color=_resolve(COLORS["card"]),
                           text_color=_resolve(COLORS["text"]),
                           corner_radius=10)
        lbl.pack(fill="both", expand=True)
        toast.after(2000, toast.destroy)

    # ------------------------------------------------------------------
    #  Properties panel — enhanced with tooltips + validation
    # ------------------------------------------------------------------

    def _update_properties(self):
        """Rebuild the properties panel for the selected layer."""
        for w in self.props_container.winfo_children():
            w.destroy()

        if self.architecture is None:
            return

        if self.selected_idx is None:
            self.props_title.configure(text="Layer Properties")
            ctk.CTkLabel(self.props_container,
                         text="Click a layer block on the\ncanvas to edit its properties.",
                         font=("Segoe UI", 12), text_color=COLORS["text_dim"],
                         justify="center").pack(expand=True, pady=20)
            return

        if self.selected_idx == 0:
            self.props_title.configure(text="📥  INPUT")
            self._build_input_props()
            return

        layer_idx = self.selected_idx - 1
        layers = self.architecture.layers
        if layer_idx < 0 or layer_idx >= len(layers):
            return

        layer = layers[layer_idx]
        self.props_title.configure(text=f"⚙  {layer.display_name}")

        ltype = layer.layer_type
        if ltype in ("conv3d_r21d", "conv3d_plain"):
            self._build_conv_props(layer)
        elif ltype == "maxpool3d":
            self._build_pool_props(layer)
        elif ltype == "dropout":
            self._build_dropout_props(layer)
        elif ltype == "batchnorm3d":
            self._build_info_props(
                "BatchNorm3D normalises activations across the batch dimension.\n\n"
                "This stabilises training, allowing higher learning rates.\n"
                "No user-editable parameters — channels are inferred automatically.")
        elif ltype == "gap3d":
            self._build_info_props(
                "Global Average Pooling 3D reduces each feature map to a single value.\n\n"
                "Converts volumetric data (C×T×H×W) to a flat vector (C,).\n"
                "Must come before the Dense (classifier) layer.")
        elif ltype == "dense":
            self._build_dense_props(layer)
        else:
            self._build_info_props(f"Layer type '{ltype}' has no editable parameters.")

        # Position info
        ctk.CTkFrame(self.props_container, height=1,
                     fg_color=COLORS["border"]).pack(fill="x", pady=(12, 6))
        pos_text = f"Position {self.selected_idx} of {len(layers)}"
        ctk.CTkLabel(self.props_container, text=pos_text,
                     font=("Segoe UI", 9), text_color=COLORS["text_dim"]).pack(anchor="w")

    # -- Property builders with tooltips --

    def _prop_label(self, text):
        ctk.CTkLabel(self.props_container, text=text,
                     font=("Segoe UI", 10, "bold"),
                     text_color=COLORS["text_dim"]).pack(anchor="w", pady=(8, 2))

    def _prop_tooltip(self, key):
        """Add a tooltip description below a property."""
        tip = _TOOLTIPS.get(key, "")
        if tip:
            ctk.CTkLabel(self.props_container, text=tip,
                         font=("Segoe UI", 9), text_color=_resolve(COLORS["text_dim"]),
                         wraplength=260, justify="left").pack(anchor="w", pady=(0, 2))

    def _prop_range(self, key):
        """Add a recommended range hint."""
        rng = _RANGES.get(key, "")
        if rng:
            ctk.CTkLabel(self.props_container, text=f"💡 {rng}",
                         font=("Segoe UI", 9, "italic"),
                         text_color=_resolve(COLORS["text_dim"]),
                         wraplength=260, justify="left").pack(anchor="w", pady=(0, 4))

    def _build_input_props(self):
        self._build_info_props(
            "Input tensor shape — determined by the\nvideo preprocessing settings.\n\n"
            "  Channels: 3 (RGB)\n"
            "  Frames: 10 per video\n"
            "  Height: 224 px\n"
            "  Width: 224 px\n\n"
            "This block cannot be deleted or moved.\n"
            "Change these in Training Configuration.")

    def _build_conv_props(self, layer):
        params = layer.params

        # Filters
        self._prop_label("FILTERS")
        self._prop_tooltip("filters")

        filter_frame = ctk.CTkFrame(self.props_container, fg_color="transparent")
        filter_frame.pack(fill="x")

        current_filters = params.get("filters", 32)
        filter_var = ctk.IntVar(value=current_filters)
        val_lbl = ctk.CTkLabel(filter_frame, text=str(current_filters),
                               font=("Segoe UI", 14, "bold"),
                               text_color=COLORS["accent"], width=40)
        val_lbl.pack(side="right")

        def on_filter_change(value):
            v = int(round(value / 8) * 8)
            if v < 8:
                v = 8
            filter_var.set(v)
            val_lbl.configure(text=str(v))
            params["filters"] = v
            self._on_param_change()

        ctk.CTkSlider(filter_frame, from_=8, to=512, number_of_steps=63,
                      variable=filter_var,
                      progress_color=COLORS["accent"],
                      button_color=COLORS["accent"],
                      button_hover_color=COLORS["accent_hover"],
                      command=on_filter_change).pack(side="left", fill="x", expand=True, padx=(0, 8))

        self._prop_range("filters")

        # Kernel size
        self._prop_label("KERNEL SIZE")
        self._prop_tooltip("kernel_size")
        k = params.get("kernel_size", 3)
        kernel_var = ctk.StringVar(value=str(k))

        def on_kernel_change(value):
            params["kernel_size"] = int(value)
            self._on_param_change()

        ctk.CTkSegmentedButton(
            self.props_container, values=["3", "5", "7"],
            variable=kernel_var, command=on_kernel_change,
            font=("Segoe UI", 11), fg_color=COLORS["input_bg"],
            selected_color=COLORS["accent"],
            selected_hover_color=COLORS["accent_hover"],
        ).pack(fill="x")

        self._prop_range("kernel_size")

        # Stride
        self._prop_label("STRIDE")
        self._prop_tooltip("stride")
        s = params.get("stride", 1)
        stride_var = ctk.StringVar(value=str(s))

        def on_stride_change(value):
            params["stride"] = int(value)
            self._on_param_change()

        ctk.CTkSegmentedButton(
            self.props_container, values=["1", "2"],
            variable=stride_var, command=on_stride_change,
            font=("Segoe UI", 11), fg_color=COLORS["input_bg"],
            selected_color=COLORS["accent"],
            selected_hover_color=COLORS["accent_hover"],
        ).pack(fill="x")

    def _build_pool_props(self, layer):
        params = layer.params
        self._prop_label("POOL SIZE")
        self._prop_tooltip("pool_size")
        p = params.get("pool_size", (2, 2, 2))
        pool_str = "×".join(str(x) for x in p) if isinstance(p, (list, tuple)) else str(p)
        pool_var = ctk.StringVar(value=pool_str if pool_str in ("2×2×2", "3×3×3") else "2×2×2")

        def on_pool_change(value):
            size = int(value[0])
            params["pool_size"] = (size, size, size)
            self._on_param_change()

        ctk.CTkSegmentedButton(
            self.props_container, values=["2×2×2", "3×3×3"],
            variable=pool_var, command=on_pool_change,
            font=("Segoe UI", 11), fg_color=COLORS["input_bg"],
            selected_color=COLORS["accent"],
            selected_hover_color=COLORS["accent_hover"],
        ).pack(fill="x")

        # Validation warning for large pool sizes
        self._prop_range("pool_size")

    def _build_dropout_props(self, layer):
        params = layer.params
        self._prop_label("DROPOUT RATE")
        self._prop_tooltip("rate")

        rate_frame = ctk.CTkFrame(self.props_container, fg_color="transparent")
        rate_frame.pack(fill="x")

        current_rate = params.get("rate", 0.3)
        rate_var = ctk.DoubleVar(value=current_rate)
        val_lbl = ctk.CTkLabel(rate_frame, text=f"{current_rate:.2f}",
                               font=("Segoe UI", 14, "bold"),
                               text_color=COLORS["accent"], width=40)
        val_lbl.pack(side="right")

        # Warning label for high dropout
        warn_lbl = ctk.CTkLabel(self.props_container, text="",
                                font=("Segoe UI", 9, "bold"),
                                text_color=_resolve(COLORS["error"]))

        def on_rate_change(value):
            v = round(value, 2)
            rate_var.set(v)
            val_lbl.configure(text=f"{v:.2f}")
            params["rate"] = v
            if v >= 0.7:
                warn_lbl.configure(text=f"⚠ Rate {v:.0%} is very high — may prevent learning")
                warn_lbl.pack(fill="x", pady=(2, 0))
            elif v >= 0.5:
                warn_lbl.configure(text=f"⚠ Rate {v:.0%} is aggressive — use with caution")
                warn_lbl.pack(fill="x", pady=(2, 0))
            else:
                warn_lbl.pack_forget()
            self._on_param_change()

        ctk.CTkSlider(rate_frame, from_=0.0, to=0.8, number_of_steps=16,
                      variable=rate_var,
                      progress_color=COLORS["accent"],
                      button_color=COLORS["accent"],
                      button_hover_color=COLORS["accent_hover"],
                      command=on_rate_change).pack(side="left", fill="x", expand=True, padx=(0, 8))

        warn_lbl.pack(fill="x", pady=(2, 0))
        # Set initial warning state
        if current_rate >= 0.7:
            warn_lbl.configure(text=f"⚠ Rate {current_rate:.0%} is very high — may prevent learning")
        elif current_rate >= 0.5:
            warn_lbl.configure(text=f"⚠ Rate {current_rate:.0%} is aggressive — use with caution")
        else:
            warn_lbl.pack_forget()

        self._prop_range("rate")

    def _build_dense_props(self, layer):
        params = layer.params
        self._prop_label("UNITS")
        self._prop_tooltip("units")

        units_var = ctk.StringVar(value=str(params.get("units", 128)))

        def on_units_change(*_args):
            try:
                v = int(units_var.get())
                if v > 0:
                    params["units"] = v
                    self._on_param_change()
            except ValueError:
                pass

        entry = ctk.CTkEntry(self.props_container, textvariable=units_var,
                             font=("Segoe UI", 12),
                             fg_color=COLORS["input_bg"],
                             border_color=COLORS["input_border"],
                             corner_radius=8)
        entry.pack(fill="x")
        entry.bind("<Return>", on_units_change)
        entry.bind("<FocusOut>", on_units_change)

        self._prop_range("units")

        ctk.CTkLabel(self.props_container,
                     text="ℹ  The final Dense layer's units will be\n"
                          "overridden by the number of classes\n"
                          "during training.",
                     font=("Segoe UI", 10), text_color=COLORS["text_dim"],
                     justify="left").pack(anchor="w", pady=(8, 0))

    def _build_info_props(self, text: str):
        ctk.CTkLabel(self.props_container, text=text,
                     font=("Segoe UI", 11), text_color=COLORS["text_dim"],
                     justify="left", wraplength=260).pack(anchor="w", pady=(6, 0))

    # ------------------------------------------------------------------
    #  Live Model Summary Table
    # ------------------------------------------------------------------

    def _update_summary_table(self):
        """Rebuild the summary table with layer shapes and param counts."""
        for w in self.summary_container.winfo_children():
            w.destroy()

        if self.architecture is None:
            return

        # Table header
        hdr = ctk.CTkFrame(self.summary_container, fg_color=_resolve(COLORS["input_bg"]),
                           corner_radius=6)
        hdr.pack(fill="x", pady=(0, 2))
        hdr.grid_columnconfigure(0, weight=3)
        hdr.grid_columnconfigure(1, weight=3)
        hdr.grid_columnconfigure(2, weight=2)

        for col, text in enumerate(["Layer", "Output Shape", "Params"]):
            ctk.CTkLabel(hdr, text=text, font=("Segoe UI", 9, "bold"),
                         text_color=_resolve(COLORS["text_dim"]),
                         anchor="w").grid(row=0, column=col, sticky="ew", padx=6, pady=4)

        # Compute shapes
        try:
            shapes = compute_shapes(self.architecture)
        except Exception:
            shapes = [None] * (len(self.architecture.layers) + 1)

        # Try to build model and get per-layer param counts
        per_layer_params = []
        total_params = 0
        try:
            model = build_from_config(self.architecture, num_classes=101)
            for module in model.features:
                p = sum(x.numel() for x in module.parameters())
                per_layer_params.append(p)
                total_params += p
        except Exception:
            per_layer_params = []

        # Rows
        layers = self.architecture.layers
        all_items = [("Input", 0)] + [(l.display_name, i + 1) for i, l in enumerate(layers)]
        param_idx = 0  # index into per_layer_params (skips input)

        for name, shape_idx in all_items:
            row = ctk.CTkFrame(self.summary_container, fg_color="transparent")
            row.pack(fill="x")
            row.grid_columnconfigure(0, weight=3)
            row.grid_columnconfigure(1, weight=3)
            row.grid_columnconfigure(2, weight=2)

            # Layer name
            layer = layers[shape_idx - 1] if shape_idx > 0 else None
            color = layer.color if layer else "#4b5563"
            name_frame = ctk.CTkFrame(row, fg_color="transparent")
            name_frame.grid(row=0, column=0, sticky="ew", padx=2, pady=1)
            ctk.CTkFrame(name_frame, width=4, height=14, fg_color=color,
                         corner_radius=2).pack(side="left", padx=(4, 4))
            ctk.CTkLabel(name_frame, text=name, font=("Segoe UI", 9),
                         text_color=_resolve(COLORS["text"]),
                         anchor="w").pack(side="left")

            # Shape
            shape = shapes[shape_idx] if shape_idx < len(shapes) else None
            shape_text = "×".join(str(s) for s in shape) if shape else "—"
            ctk.CTkLabel(row, text=shape_text, font=("Segoe UI", 9),
                         text_color=_resolve(COLORS["text_dim"]),
                         anchor="w").grid(row=0, column=1, sticky="ew", padx=6, pady=1)

            # Params
            if shape_idx == 0:
                param_text = "—"
            elif per_layer_params and param_idx < len(per_layer_params):
                # Skip "input" layers (they don't produce a module)
                if layer and layer.layer_type != "input":
                    p = per_layer_params[param_idx]
                    param_text = f"{p:,}" if p > 0 else "0"
                    param_idx += 1
                else:
                    param_text = "—"
            else:
                param_text = "—"

            ctk.CTkLabel(row, text=param_text, font=("Segoe UI", 9),
                         text_color=_resolve(COLORS["text_dim"]),
                         anchor="w").grid(row=0, column=2, sticky="ew", padx=6, pady=1)

        # Total row
        ctk.CTkFrame(self.summary_container, height=1,
                     fg_color=COLORS["border"]).pack(fill="x", pady=(4, 2))
        total_row = ctk.CTkFrame(self.summary_container, fg_color="transparent")
        total_row.pack(fill="x")
        total_row.grid_columnconfigure(0, weight=3)
        total_row.grid_columnconfigure(1, weight=3)
        total_row.grid_columnconfigure(2, weight=2)
        ctk.CTkLabel(total_row, text="TOTAL", font=("Segoe UI", 9, "bold"),
                     text_color=_resolve(COLORS["text"]),
                     anchor="w").grid(row=0, column=0, sticky="ew", padx=6, pady=2)
        ctk.CTkLabel(total_row, text="", font=("Segoe UI", 9),
                     text_color=_resolve(COLORS["text_dim"])).grid(row=0, column=1)
        ctk.CTkLabel(total_row, text=f"{total_params:,}" if total_params else "—",
                     font=("Segoe UI", 9, "bold"),
                     text_color=_resolve(COLORS["accent"]),
                     anchor="w").grid(row=0, column=2, sticky="ew", padx=6, pady=2)

        # Validation warnings
        if self._warnings:
            ctk.CTkFrame(self.summary_container, height=1,
                         fg_color=COLORS["border"]).pack(fill="x", pady=(6, 4))
            for warn in self._warnings:
                ctk.CTkLabel(self.summary_container, text=f"⚠ {warn}",
                             font=("Segoe UI", 9),
                             text_color=_resolve(COLORS["error"]),
                             wraplength=280, justify="left",
                             anchor="w").pack(fill="x", padx=4, pady=1)

    # ------------------------------------------------------------------
    #  Parameter change handler
    # ------------------------------------------------------------------

    def _on_param_change(self):
        """Called whenever a layer parameter is changed by the user."""
        self._redraw_canvas()
        self._update_summary_table()
        self._update_status()

    # ------------------------------------------------------------------
    #  Status bar update
    # ------------------------------------------------------------------

    def _update_status(self):
        if self.architecture is None:
            return

        n_layers = len(self.architecture.layers)

        try:
            model = build_from_config(self.architecture, num_classes=101)
            total, trainable = count_params(model)
            self.status_params.configure(
                text=f"Parameters: {total:,}  ({trainable:,} trainable)")
        except Exception:
            self.status_params.configure(text="Parameters: –  (build error)")

        # Warnings
        try:
            self._warnings = validate_architecture(self.architecture)
        except Exception:
            self._warnings = []

        if self._warnings:
            self.status_warnings.configure(
                text=f"⚠ {len(self._warnings)} warning{'s' if len(self._warnings) > 1 else ''}")
        else:
            self.status_warnings.configure(text="✓ Valid", text_color=COLORS["success"])

        name = self.architecture.name or "Unnamed"
        self.status_name.configure(text=name)
        self.status_layers.configure(text=f"Layers: {n_layers}")

    def on_theme_changed(self, mode):
        self.configure(fg_color=COLORS["bg"])
        # Update canvas bg
        self.canvas.configure(bg=_resolve(COLORS["console_bg"]))
        # Trigger full canvas redraw with correct mode-resolved colors
        self._redraw_canvas()
