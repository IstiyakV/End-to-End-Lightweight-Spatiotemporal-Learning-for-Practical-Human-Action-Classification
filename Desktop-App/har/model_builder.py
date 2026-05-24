"""
Model Builder — visual architecture → PyTorch model.

Defines layer configurations and builds nn.Module models from
user-defined layer sequences created in the Network Architect GUI.

Public API
----------
- ``default_architecture()`` — paper's Compact3DCNN expressed as config
- ``build_from_config()``    — ArchitectureConfig → nn.Module
- ``compute_shapes()``       — analytic shape propagation (no forward pass)
- ``count_params()``         — total / trainable parameter counts
- ``list_saved_architectures()`` — enumerate saved ``.json`` files
"""

import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn

from . import config
from .model import SpatioTemporalResBlock

# ── Layer Type Registry ──────────────────────────────────────────────────────

LAYER_TYPES: Dict[str, str] = {
    "input":        "Input",
    "conv3d_r21d":  "Conv3D R(2+1)D",
    "conv3d_plain": "Conv3D",
    "maxpool3d":    "MaxPool3D",
    "batchnorm3d":  "BatchNorm3D",
    "dropout":      "Dropout",
    "gap3d":        "Global Avg Pool",
    "dense":        "Dense",
}

# Colours for canvas blocks (dark-mode hex palette)
LAYER_COLORS: Dict[str, str] = {
    "input":        "#636e72",
    "conv3d_r21d":  "#6c5ce7",
    "conv3d_plain": "#0984e3",
    "maxpool3d":    "#00b894",
    "batchnorm3d":  "#fdcb6e",
    "dropout":      "#e17055",
    "gap3d":        "#00cec9",
    "dense":        "#d63031",
}


# ── Per-layer default parameters ─────────────────────────────────────────────

def get_default_params(layer_type: str) -> dict:
    """Return sensible default parameters for a newly-created layer.

    Parameters
    ----------
    layer_type : str
        One of the keys in :data:`LAYER_TYPES`.

    Returns
    -------
    dict
        Shallow copy of the defaults so callers can mutate freely.
    """
    _DEFAULTS: Dict[str, dict] = {
        "input":        {"channels": 3},
        "conv3d_r21d":  {"filters": 64, "kernel_size": 3, "stride": 1},
        "conv3d_plain": {"filters": 64, "kernel_size": 3, "stride": 1, "padding": 1},
        "maxpool3d":    {"pool_size": [2, 2, 2]},
        "batchnorm3d":  {},
        "dropout":      {"rate": 0.3},
        "gap3d":        {},
        "dense":        {"units": 128},
    }
    return dict(_DEFAULTS.get(layer_type, {}))


# ── Layer / Architecture configs ─────────────────────────────────────────────

@dataclass
class LayerConfig:
    """Configuration for a single network layer."""

    layer_type: str
    params: dict = field(default_factory=dict)

    # ── convenience properties ──

    @property
    def display_name(self) -> str:
        """Human-readable name shown in the GUI palette."""
        return LAYER_TYPES.get(self.layer_type, self.layer_type)

    @property
    def color(self) -> str:
        """Hex colour used for the canvas block."""
        return LAYER_COLORS.get(self.layer_type, "#636e72")

    @property
    def summary(self) -> str:
        """Short string for the canvas label, e.g. ``'64 ch'`` or ``'p=0.3'``."""
        lt = self.layer_type
        p = self.params

        if lt == "input":
            ch = p.get("channels", 3)
            return f"{ch} ch"
        if lt in ("conv3d_r21d", "conv3d_plain"):
            f = p.get("filters", "?")
            k = p.get("kernel_size", 3)
            return f"{f} ch, k={k}"
        if lt == "maxpool3d":
            ps = p.get("pool_size", [2, 2, 2])
            if isinstance(ps, (list, tuple)):
                return "×".join(str(s) for s in ps)
            return str(ps)
        if lt == "batchnorm3d":
            return "BN"
        if lt == "dropout":
            return f"p={p.get('rate', 0.3)}"
        if lt == "gap3d":
            return "GAP"
        if lt == "dense":
            return f"{p.get('units', '?')} units"
        return ""


@dataclass
class ArchitectureConfig:
    """Full network architecture as an ordered list of layers.

    Serialisable to / from JSON for persistence.
    """

    name: str = "Custom"
    layers: List[LayerConfig] = field(default_factory=list)

    # ── serialisation ──

    def to_dict(self) -> dict:
        """Convert to a plain ``dict`` suitable for :func:`json.dumps`."""
        return {
            "name": self.name,
            "layers": [
                {"layer_type": lc.layer_type, "params": lc.params}
                for lc in self.layers
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ArchitectureConfig":
        """Reconstruct from a ``dict`` produced by :meth:`to_dict`.

        Parameters
        ----------
        d : dict
            Must contain ``"name"`` (str) and ``"layers"`` (list of dicts
            each with ``"layer_type"`` and ``"params"``).

        Returns
        -------
        ArchitectureConfig
        """
        layers = [
            LayerConfig(
                layer_type=ld["layer_type"],
                params=ld.get("params", {}),
            )
            for ld in d.get("layers", [])
        ]
        return cls(name=d.get("name", "Custom"), layers=layers)

    def save(self, path: Path) -> None:
        """Write the architecture to a JSON file.

        Parameters
        ----------
        path : Path
            Destination file (created / overwritten).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "ArchitectureConfig":
        """Load an architecture from a JSON file.

        Parameters
        ----------
        path : Path
            Source ``.json`` file.

        Returns
        -------
        ArchitectureConfig

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        json.JSONDecodeError
            If the file contains invalid JSON.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# ── Default (paper) architecture ─────────────────────────────────────────────

def default_architecture() -> ArchitectureConfig:
    """Return the paper's Compact3DCNN expressed as an :class:`ArchitectureConfig`.

    Structure::

        Input(3)
        → Conv3D_R21D(32)  → MaxPool3D(2×2×2)
        → Conv3D_R21D(64)  → MaxPool3D(2×2×2)
        → Conv3D_R21D(128) → MaxPool3D(2×2×2)
        → Dropout(0.3)
        → GAP3D
        → Dense(num_classes)          ← units are set at build time

    The final Dense layer uses a sentinel ``units=0`` so that
    :func:`build_from_config` replaces it with the actual *num_classes*.
    """
    layers: List[LayerConfig] = [
        LayerConfig("input", {"channels": 3}),
    ]

    # Three R(2+1)D conv blocks, each followed by MaxPool3D
    for filters in config.CONV_FILTERS:          # [32, 64, 128]
        layers.append(LayerConfig("conv3d_r21d", {
            "filters": filters,
            "kernel_size": config.KERNEL_SIZE,    # 3
            "stride": 1,
        }))
        layers.append(LayerConfig("maxpool3d", {
            "pool_size": list(config.POOL_SIZE),  # [2, 2, 2]
        }))

    layers.append(LayerConfig("dropout", {"rate": config.DROPOUT}))
    layers.append(LayerConfig("gap3d", {}))
    # units=0 → replaced by num_classes at build time
    layers.append(LayerConfig("dense", {"units": 0}))

    return ArchitectureConfig(name="Compact3DCNN (Paper)", layers=layers)


# ── Analytic shape propagation ───────────────────────────────────────────────

def _pool_dim(size: int, kernel: int) -> int:
    """Floor-division pooling: ``size // kernel``."""
    return size // kernel


def compute_shapes(
    arch: ArchitectureConfig,
    input_shape: Optional[Tuple[int, ...]] = None,
) -> List[Tuple[int, ...]]:
    """Propagate tensor shapes through the architecture analytically.

    No forward pass is performed — shapes are computed from convolution /
    pooling arithmetic.

    Parameters
    ----------
    arch : ArchitectureConfig
        Architecture definition.
    input_shape : tuple of int, optional
        ``(C, T, H, W)``.  Defaults to
        ``(3, config.N_FRAMES, *config.IMG_SIZE)``.

    Returns
    -------
    list of tuple
        One shape per layer (including the initial input).  Shapes are
        ``(C, T, H, W)`` for volumetric layers or ``(features,)`` for
        dense / post-GAP layers.
    """
    if input_shape is None:
        input_shape = (3, config.N_FRAMES, config.IMG_SIZE[0], config.IMG_SIZE[1])

    C, T, H, W = input_shape
    shapes: List[Tuple[int, ...]] = [(C, T, H, W)]
    flattened = False  # track whether we've collapsed spatial dims

    for lc in arch.layers:
        lt = lc.layer_type
        p = lc.params

        if lt == "input":
            # Already captured above; nothing changes.
            shapes.append((C, T, H, W))

        elif lt in ("conv3d_r21d", "conv3d_plain"):
            filters = p.get("filters", 64)
            stride = p.get("stride", 1)

            if lt == "conv3d_r21d":
                # R(2+1)D block: spatial conv (1×3×3, pad=(0,1,1), stride=(1,s,s))
                # followed by temporal conv (3×1×1, pad=(1,0,0), stride=(s,1,1))
                # Net effect with stride=1 and the existing padding: same T, H, W
                # With stride>1: H, W halved by spatial stride; T halved by temporal stride
                T_out = math.floor((T + 2 * 1 - 3) / stride + 1)  # temporal conv
                H_out = math.floor((H + 2 * 1 - 3) / stride + 1)  # spatial conv
                W_out = math.floor((W + 2 * 1 - 3) / stride + 1)
            else:
                # Plain Conv3d with symmetric kernel and padding
                k = p.get("kernel_size", 3)
                pad = p.get("padding", 1)
                T_out = math.floor((T + 2 * pad - k) / stride + 1)
                H_out = math.floor((H + 2 * pad - k) / stride + 1)
                W_out = math.floor((W + 2 * pad - k) / stride + 1)

            C, T, H, W = filters, T_out, H_out, W_out
            shapes.append((C, T, H, W))

        elif lt == "maxpool3d":
            ps = p.get("pool_size", [2, 2, 2])
            if isinstance(ps, int):
                ps = [ps, ps, ps]
            T = _pool_dim(T, ps[0])
            H = _pool_dim(H, ps[1])
            W = _pool_dim(W, ps[2])
            shapes.append((C, T, H, W))

        elif lt == "batchnorm3d":
            # Shape unchanged
            shapes.append((C, T, H, W) if not flattened else shapes[-1])

        elif lt == "dropout":
            # Shape unchanged
            shapes.append((C, T, H, W) if not flattened else shapes[-1])

        elif lt == "gap3d":
            # AdaptiveAvgPool3d(1) → (C, 1, 1, 1) → flatten → (C,)
            flattened = True
            shapes.append((C,))

        elif lt == "dense":
            units = p.get("units", 128)
            C = units  # reuse C for feature dim after flatten
            flattened = True
            shapes.append((units,))

        else:
            # Unknown layer type — carry shape forward unchanged
            shapes.append(shapes[-1] if shapes else (C, T, H, W))

    return shapes


# ── Dynamic model construction ───────────────────────────────────────────────

class DynamicModel(nn.Module):
    """A PyTorch model built dynamically from an :class:`ArchitectureConfig`.

    The constructor translates each :class:`LayerConfig` into real
    ``nn.Module`` layers and stores them in an ``nn.ModuleList`` so that
    parameters are correctly registered.

    Parameters
    ----------
    arch : ArchitectureConfig
        Architecture definition.
    num_classes : int
        Number of output classes.  The *last* ``dense`` layer's units are
        overridden with this value.
    dropout : float
        Default dropout probability (used when a dropout layer's config
        does not specify ``rate``).
    """

    def __init__(
        self,
        arch: ArchitectureConfig,
        num_classes: int,
        dropout: float = config.DROPOUT,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes

        # We need to know the channel count flowing into each layer.
        in_channels = 3  # default; overridden by input layer's config

        modules: List[nn.Module] = []
        last_conv_channels = in_channels  # tracks C for the classifier

        # Find index of last dense layer so we can override its units
        last_dense_idx = -1
        for i, lc in enumerate(arch.layers):
            if lc.layer_type == "dense":
                last_dense_idx = i

        for i, lc in enumerate(arch.layers):
            lt = lc.layer_type
            p = lc.params

            if lt == "input":
                in_channels = p.get("channels", 3)
                continue  # no module to add

            elif lt == "conv3d_r21d":
                out_ch = p.get("filters", 64)
                stride = p.get("stride", 1)
                modules.append(SpatioTemporalResBlock(in_channels, out_ch, stride=stride))
                in_channels = out_ch
                last_conv_channels = out_ch

            elif lt == "conv3d_plain":
                out_ch = p.get("filters", 64)
                k = p.get("kernel_size", 3)
                stride = p.get("stride", 1)
                pad = p.get("padding", 1)
                modules.append(nn.Sequential(
                    nn.Conv3d(in_channels, out_ch, kernel_size=k,
                              stride=stride, padding=pad, bias=True),
                    nn.ReLU(inplace=False),
                ))
                in_channels = out_ch
                last_conv_channels = out_ch

            elif lt == "maxpool3d":
                ps = p.get("pool_size", [2, 2, 2])
                if isinstance(ps, int):
                    ps = (ps, ps, ps)
                else:
                    ps = tuple(ps)
                modules.append(nn.MaxPool3d(kernel_size=ps))

            elif lt == "batchnorm3d":
                modules.append(nn.BatchNorm3d(in_channels))

            elif lt == "dropout":
                rate = p.get("rate", dropout)
                modules.append(nn.Dropout(rate))

            elif lt == "gap3d":
                modules.append(nn.AdaptiveAvgPool3d(1))

            elif lt == "dense":
                units = p.get("units", 128)
                # Override the last dense layer's units with num_classes
                if i == last_dense_idx:
                    units = num_classes
                modules.append(nn.Linear(in_channels, units))
                in_channels = units

        self.features = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dynamic layer stack.

        Automatically flattens the tensor when transitioning from a 5-D
        (volumetric) representation to a 2-D (batch × features) one, i.e.
        just before the first ``nn.Linear`` layer.
        """
        for module in self.features:
            if isinstance(module, nn.Linear):
                # Flatten if we still have spatial dims
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
            x = module(x)
        return x


def build_from_config(
    arch: ArchitectureConfig,
    num_classes: int,
    dropout: float = config.DROPOUT,
) -> nn.Module:
    """Build a real PyTorch :class:`nn.Module` from an architecture config.

    When given the architecture returned by :func:`default_architecture`,
    this produces a model functionally identical to
    :class:`~har.model.Compact3DCNN`.

    Parameters
    ----------
    arch : ArchitectureConfig
        Architecture definition (ordered layer list).
    num_classes : int
        Number of output classes.  The final dense layer's ``units`` are
        overridden with this value regardless of the config.
    dropout : float
        Default dropout probability.

    Returns
    -------
    nn.Module
        A :class:`DynamicModel` instance ready for training.
    """
    return DynamicModel(arch, num_classes=num_classes, dropout=dropout)


# ── Preset architectures ─────────────────────────────────────────────────────

def _make_preset(
    name: str,
    filters: List[int],
    use_r21d: bool = True,
    dropout: float = 0.3,
    pool_size: List[int] = None,
) -> ArchitectureConfig:
    """Build a preset architecture from a compact spec."""
    if pool_size is None:
        pool_size = [2, 2, 2]
    conv_type = "conv3d_r21d" if use_r21d else "conv3d_plain"
    layers: List[LayerConfig] = [LayerConfig("input", {"channels": 3})]
    for f in filters:
        params = {"filters": f, "kernel_size": 3, "stride": 1}
        if not use_r21d:
            params["padding"] = 1
        layers.append(LayerConfig(conv_type, params))
        layers.append(LayerConfig("maxpool3d", {"pool_size": list(pool_size)}))
    layers.append(LayerConfig("dropout", {"rate": dropout}))
    layers.append(LayerConfig("gap3d", {}))
    layers.append(LayerConfig("dense", {"units": 0}))
    return ArchitectureConfig(name=name, layers=layers)


PRESET_META: Dict[str, Dict[str, Any]] = {
    "Paper Default": {
        "description": "Compact R(2+1)D from your paper §III-C",
        "badge": "Recommended",
        "badge_color": "#00b894",
    },
    "Lightweight": {
        "description": "2 layers, tiny filters — fast prototyping",
        "badge": "Beginner",
        "badge_color": "#0984e3",
    },
    "Deep": {
        "description": "4 conv blocks with 256 ch — high capacity",
        "badge": "Advanced",
        "badge_color": "#e17055",
    },
    "Plain Conv3D": {
        "description": "No residual connections — simple baseline",
        "badge": "Baseline",
        "badge_color": "#fdcb6e",
    },
}


def _make_deep_preset() -> ArchitectureConfig:
    """Build the Deep preset with heterogeneous pool sizes.

    With 10 input frames, 4× pool [2,2,2] collapses T to 0.
    The 4th pool uses [1,2,2] (spatial-only) to keep T=1.
    """
    filters = [32, 64, 128, 256]
    pool_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
    layers: List[LayerConfig] = [LayerConfig("input", {"channels": 3})]
    for f, ps in zip(filters, pool_sizes):
        layers.append(LayerConfig("conv3d_r21d", {
            "filters": f, "kernel_size": 3, "stride": 1,
        }))
        layers.append(LayerConfig("maxpool3d", {"pool_size": ps}))
    layers.append(LayerConfig("dropout", {"rate": 0.4}))
    layers.append(LayerConfig("gap3d", {}))
    layers.append(LayerConfig("dense", {"units": 0}))
    return ArchitectureConfig(name="Deep", layers=layers)


def preset_architectures() -> Dict[str, ArchitectureConfig]:
    """Return a dictionary of named architecture presets.

    Keys match :data:`PRESET_META` for UI badge/description lookup.
    """
    return {
        "Paper Default": _make_preset(
            "Paper Default",
            filters=list(config.CONV_FILTERS),  # [32, 64, 128]
            use_r21d=True,
            dropout=config.DROPOUT,
        ),
        "Lightweight": _make_preset(
            "Lightweight",
            filters=[16, 32],
            use_r21d=True,
            dropout=0.2,
        ),
        "Deep": _make_deep_preset(),
        "Plain Conv3D": _make_preset(
            "Plain Conv3D",
            filters=[32, 64, 128],
            use_r21d=False,
            dropout=0.3,
        ),
    }


# ── Architecture validation ──────────────────────────────────────────────────

def validate_architecture(arch: ArchitectureConfig) -> List[str]:
    """Check an architecture for common mistakes.

    Returns a list of human-readable warning strings (empty = all OK).
    """
    warnings: List[str] = []
    layers = arch.layers

    if not layers:
        warnings.append("Architecture has no layers.")
        return warnings

    # Must start with input
    if layers[0].layer_type != "input":
        warnings.append("First layer should be 'input'.")

    # Must have at least one conv layer
    conv_types = {"conv3d_r21d", "conv3d_plain"}
    has_conv = any(l.layer_type in conv_types for l in layers)
    if not has_conv:
        warnings.append("No convolutional layers — the model cannot learn spatial features.")

    # Must have GAP before Dense
    has_gap = any(l.layer_type == "gap3d" for l in layers)
    has_dense = any(l.layer_type == "dense" for l in layers)
    if has_dense and not has_gap:
        warnings.append("Dense layer without GAP — spatial dimensions will be flattened incorrectly.")

    if not has_dense:
        warnings.append("No output Dense layer — the model has no classifier head.")

    # Check GAP comes before Dense
    gap_idx = next((i for i, l in enumerate(layers) if l.layer_type == "gap3d"), None)
    dense_idx = next((i for i, l in enumerate(layers) if l.layer_type == "dense"), None)
    if gap_idx is not None and dense_idx is not None and gap_idx > dense_idx:
        warnings.append("GAP layer appears after Dense — it should come before.")

    # Check for dimension collapse via shape propagation
    try:
        shapes = compute_shapes(arch)
        for i, shape in enumerate(shapes):
            if len(shape) == 4:
                C, T, H, W = shape
                if T <= 0 or H <= 0 or W <= 0:
                    layer_name = layers[i].display_name if i < len(layers) else "output"
                    warnings.append(
                        f"Dimension collapsed to zero at layer {i} ({layer_name}). "
                        f"Reduce pooling or add fewer pool layers."
                    )
                    break
    except Exception:
        warnings.append("Could not compute output shapes — architecture may be invalid.")

    # Check for very high dropout
    for l in layers:
        if l.layer_type == "dropout":
            rate = l.params.get("rate", 0.3)
            if rate >= 0.7:
                warnings.append(f"Dropout rate {rate:.0%} is very high — may prevent learning.")

    return warnings


# ── Utilities ────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> Tuple[int, int]:
    """Count model parameters.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch module.

    Returns
    -------
    total : int
        Total number of parameters.
    trainable : int
        Number of parameters with ``requires_grad=True``.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ── Persistence helpers ──────────────────────────────────────────────────────

ARCH_DIR: Path = config.RESULTS_DIR / "architectures"
ARCH_DIR.mkdir(parents=True, exist_ok=True)


def list_saved_architectures() -> List[str]:
    """List all saved architecture names (without the ``.json`` extension).

    Scans :data:`ARCH_DIR` for ``*.json`` files.

    Returns
    -------
    list of str
        Sorted architecture names.
    """
    if not ARCH_DIR.exists():
        return []
    return sorted(p.stem for p in ARCH_DIR.glob("*.json"))
