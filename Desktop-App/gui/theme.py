"""Centralized UI Theme configuration for Light and Dark modes."""

# Colors are defined as (Light_Mode_Hex, Dark_Mode_Hex)
COLORS = {
    "bg": ("#f8fafc", "#0f1117"),          # Slate-50 off-white for dynamic backdrop
    "card": ("#ffffff", "#181b28"),        # Crisp card contrast
    "sidebar": ("#ffffff", "#151825"),     # Crisp sidebar panel
    "accent": ("#4f46e5", "#6c5ce7"),      # Premium slate-indigo instead of heavy purple
    "accent_hover": ("#4338ca", "#7c6cf7"),
    "success": ("#16a34a", "#00b894"),     # Forest-green-600 / Emerald
    "error": ("#dc2626", "#e17055"),       # Crimson / Coral
    "warning": ("#d97706", "#fdcb6e"),     # Amber / Gold
    "text": ("#0f172a", "#e4e6eb"),        # Slate-900 high contrast readable text
    "text_dim": ("#64748b", "#8b8fa3"),    # Slate-500 medium subtitle text
    "border": ("#e2e8f0", "#252836"),      # Slate-200 delicate divider line
    "card_border": ("#cbd5e1", "#2a2d42"), # Slate-300 card borders
    "input_bg": ("#f1f5f9", "#1e2235"),    # Slate-100 clean input backdrop
    "input_border": ("#cbd5e1", "#2d3048"),
    
    # Specific component shades
    "sidebar_hover": ("#f1f5f9", "#2a2d3a"),
    "progress_bg":   ("#e2e8f0", "#2a2d3a"),
    "plot_bg":       ("#ffffff", "#181b28"),
    "plot_grid":     ("#f1f5f9", "#252836"),  # Extremely subtle light gray grid
    "console_bg":    ("#f8fafc", "#0d1117"),  # Cool gray terminal console
    "gold":          ("#b45309", "#ffd700"),
}

