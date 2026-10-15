"""
Plotting helpers for Transformer Observatory.

All functions return a Matplotlib figure; the caller is responsible for closing it.

Changes relative to the original module (all backwards compatible: every new
argument is optional and the defaults reproduce the original look):

  * ``theme``       "light" (original look) or "dark" (matches the app's dark theme)
  * ``figsize``     override the figure size so panels can be compact
  * ``show_title``  allow the caller to put the title in its own panel header
  * ``annotate``    attention cell annotations are skipped automatically for long
                    prompts (auto = on for <= 20 tokens) where they are unreadable
  * Tick / bar labels now make whitespace visible (space -> "·", newline -> "\\n")
    instead of passing raw newlines into tick labels, which broke the layout.
  * ``plot_qkv``    new: Q/K/V heatmap for a single token, head and layer.
  * ``plot_attention_stages``  new: raw / scaled / masked scores and softmax for one query token.
  * ``plot_hidden_states``     new: one token's vector at every residual-stream stage (needs Matplotlib >= 3.5).
"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


ANNOTATE_MAX_TOKENS = 20

_PALETTES = {
    "light": {
        "fig_bg": "white",
        "text": "black",
        "spine": "black",
        "grid": "#cccccc",
        "attn_cmap": "viridis",
        "div_cmap": "RdBu_r",
        "bar": "tab:blue",
        "bar_top": "tab:blue",
        "highlight": "yellow",
        "on_strong": "white",
        "on_weak": "black",
        "masked": "#bbbbbb",
    },
    "dark": {
        "fig_bg": "#0e131b",
        "text": "#c5d0df",
        "spine": "#2a3647",
        "grid": "#2a3647",
        "attn_cmap": LinearSegmentedColormap.from_list(
            "obs_attn", ["#0f1722", "#134e5e", "#2dd4bf", "#e6fffb"]
        ),
        "div_cmap": LinearSegmentedColormap.from_list(
            "obs_div", ["#f59e0b", "#18212e", "#2dd4bf"]
        ),
        "bar": "#1c7f7a",
        "bar_top": "#2dd4bf",
        "highlight": "#fbbf24",
        "on_strong": "#0b0f14",
        "on_weak": "#d6dfeb",
        "masked": "#273244",
    },
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _palette(theme):
    try:
        return _PALETTES[theme]
    except KeyError:
        raise ValueError(
            f"Unknown theme {theme!r}; choose from {sorted(_PALETTES)}."
        ) from None


def _to_numpy(value):
    """Accept torch tensors (any device/dtype) or array-likes."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value)


def token_label(token):
    """Make whitespace visible: space -> '·', newline -> '\\n', tab -> '\\t'."""
    label = (
        token.replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace(" ", "·")
    )
    return label or "<empty>"


def _tick_label(token):
    # Matplotlib treats paired "$" as math text; escape so tokens render literally.
    return token_label(token).replace("$", r"\$")


def _token_fontsize(count):
    return max(5.5, min(9.0, 220.0 / max(count, 1)))


def _style_axes(ax, pal):
    ax.set_facecolor(pal["fig_bg"])
    for spine in ax.spines.values():
        spine.set_color(pal["spine"])
    ax.tick_params(colors=pal["text"])
    ax.xaxis.label.set_color(pal["text"])
    ax.yaxis.label.set_color(pal["text"])
    ax.title.set_color(pal["text"])


def _style_colorbar(cbar, label, pal, labelsize=8):
    cbar.set_label(label, color=pal["text"])
    cbar.ax.tick_params(colors=pal["text"], labelsize=labelsize)
    cbar.outline.set_edgecolor(pal["spine"])


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #

def plot_attention(
    result,
    layer_idx=0,
    head_idx=0,
    position=None,
    theme="light",
    figsize=(10, 8),
    annotate=None,
    show_title=True,
):
    pal = _palette(theme)
    attention = _to_numpy(result["attention_weights"][layer_idx][head_idx])
    tokens = result["tokens"]
    count = len(tokens)

    if position is not None and (position < 0 or position >= count):
        raise ValueError("Position is outside the prompt.")

    if annotate is None:
        annotate = count <= ANNOTATE_MAX_TOKENS

    fig, ax = plt.subplots(figsize=figsize, facecolor=pal["fig_bg"])
    _style_axes(ax, pal)

    image = ax.imshow(
        attention,
        cmap=pal["attn_cmap"],
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    labels = [_tick_label(token) for token in tokens]
    fontsize = _token_fontsize(count)

    ax.set_xticks(range(count))
    ax.set_xticklabels(
        labels, rotation=45, ha="right", fontsize=fontsize, fontfamily="monospace"
    )
    ax.set_yticks(range(count))
    ax.set_yticklabels(labels, fontsize=fontsize, fontfamily="monospace")

    if annotate:
        for i in range(count):
            for j in range(count):
                if attention[i, j] > 0:
                    ax.text(
                        j,
                        i,
                        f"{attention[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if attention[i, j] < 0.5 else "black",
                    )

    if position is not None:
        ax.add_patch(
            Rectangle(
                (-0.5, position - 0.5),
                count,
                1,
                linewidth=2.0,
                edgecolor=pal["highlight"],
                facecolor="none",
                zorder=3,
            )
        )
        # Make the selected query token easy to find on the y axis as well.
        selected = ax.get_yticklabels()[position]
        selected.set_color(pal["highlight"])
        selected.set_fontweight("bold")

    ax.set_xlabel("Key token")
    ax.set_ylabel("Query token")
    if show_title:
        ax.set_title(f"Layer {layer_idx} — Head {head_idx} Attention")

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    _style_colorbar(cbar, "Attention weight", pal)
    fig.tight_layout()

    return fig


# --------------------------------------------------------------------------- #
# Next-token predictions
# --------------------------------------------------------------------------- #

def plot_predictions(
    result,
    top_k=10,
    theme="light",
    figsize=(10, 6),
    show_title=True,
):
    pal = _palette(theme)
    predictions = result["top_predictions"][:top_k]
    labels = [_tick_label(item["token"]) for item in predictions]
    probabilities = [item["probability"] for item in predictions]
    rows = list(range(len(predictions)))

    fig, ax = plt.subplots(figsize=figsize, facecolor=pal["fig_bg"])
    _style_axes(ax, pal)

    colors = [pal["bar_top"]] + [pal["bar"]] * max(len(rows) - 1, 0)
    ax.barh(rows, probabilities, color=colors[: len(rows)], height=0.7)
    ax.set_yticks(rows)
    ax.set_yticklabels(labels, fontsize=9, fontfamily="monospace")
    ax.invert_yaxis()  # most probable token on top

    if probabilities:
        ax.set_xlim(0, max(probabilities) * 1.08)
    ax.xaxis.grid(True, color=pal["grid"], linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    ax.set_xlabel("Probability")
    ax.set_ylabel("Next token")
    if show_title:
        ax.set_title("Top Next-Token Predictions")
    fig.tight_layout()

    return fig


# --------------------------------------------------------------------------- #
# Q / K / V
# --------------------------------------------------------------------------- #

def plot_qkv(vectors, theme="light", figsize=(8, 2.4), annotate=True):
    """
    Heatmap of Query / Key / Value vectors for one token, layer and head.

    ``vectors`` maps a row label to a 1-D array of length head_dim, e.g.
    {"Query (Q)": q, "Key (K)": k, "Value (V)": v}. All rows share one symmetric
    colour scale so magnitudes are comparable across Q, K and V.
    """
    pal = _palette(theme)
    names = list(vectors)
    data = np.stack([_to_numpy(vec).reshape(-1) for vec in vectors.values()])
    rows, dim = data.shape
    vmax = float(np.abs(data).max()) or 1.0

    fig, ax = plt.subplots(figsize=figsize, facecolor=pal["fig_bg"])
    _style_axes(ax, pal)

    image = ax.imshow(
        data, cmap=pal["div_cmap"], vmin=-vmax, vmax=vmax, aspect="auto"
    )

    ax.set_xticks(range(dim))
    ax.set_xticklabels([str(i) for i in range(dim)], fontsize=7)
    ax.set_yticks(range(rows))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Head dimension")

    # Thin separators between cells.
    ax.set_xticks(np.arange(-0.5, dim, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color=pal["fig_bg"], linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    if annotate and dim <= 32:
        fontsize = float(np.clip(110.0 / dim, 4.5, 8.0))
        for r in range(rows):
            for c in range(dim):
                value = data[r, c]
                strong = abs(value) / vmax > 0.6
                ax.text(
                    c,
                    r,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=pal["on_strong"] if strong else pal["on_weak"],
                )

    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    _style_colorbar(cbar, "Value", pal, labelsize=7)
    fig.tight_layout()

    return fig


# --------------------------------------------------------------------------- #
# Attention calculation, stage by stage
# --------------------------------------------------------------------------- #

def _cmap_with_masked(spec, pal):
    """Copy of a colormap whose 'bad' (NaN) colour is the palette's masked colour."""
    cmap = matplotlib.colormaps[spec] if isinstance(spec, str) else spec
    return cmap.with_extremes(bad=pal["masked"])  # returns a copy; the shared colormap is untouched


def plot_attention_stages(
    raw,
    scaled,
    masked,
    probabilities,
    tokens,
    position,
    theme="light",
    figsize=(7, 3.4),
    annotate=None,
):
    """
    The attention calculation of one query token, stage by stage (one column per key token).

    ``raw``, ``scaled``, ``masked`` and ``probabilities`` are 1-D arrays over the keys of the
    query at ``position``; ``masked`` holds -inf for future keys. Rows, top to bottom:

        raw     Q K^T
        scaled  Q K^T / sqrt(d_k)
        mask    scaled, with future keys removed (shown as -inf)
        softmax probabilities (future keys are exactly 0)

    The first three rows share one symmetric colour scale (set by the visible keys only, so
    values at masked keys may saturate); the dashed line marks the causal boundary and the
    greyed columns are keys the query may not attend to.
    """
    pal = _palette(theme)
    raw = _to_numpy(raw).astype(float)
    scaled = _to_numpy(scaled).astype(float)
    masked = _to_numpy(masked).astype(float)
    probabilities = _to_numpy(probabilities).astype(float)
    count = len(tokens)

    if position < 0 or position >= count:
        raise ValueError("Position is outside the prompt.")

    allowed = np.arange(count) <= position
    scores = np.vstack([raw, scaled, np.where(allowed, scaled, np.nan)])
    probs_image = np.where(allowed, probabilities, np.nan)[None, :]
    vmax = float(np.abs(np.concatenate([raw[allowed], scaled[allowed]])).max()) or 1.0

    if annotate is None:
        annotate = count <= 24

    fig = plt.figure(figsize=figsize, facecolor=pal["fig_bg"], layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[50, 1])
    ax_scores = fig.add_subplot(grid[0, 0])
    cax_scores = fig.add_subplot(grid[0, 1])
    ax_probs = fig.add_subplot(grid[1, 0], sharex=ax_scores)
    cax_probs = fig.add_subplot(grid[1, 1])
    for axis in (ax_scores, ax_probs):
        _style_axes(axis, pal)

    scores_image = ax_scores.imshow(
        scores, cmap=_cmap_with_masked(pal["div_cmap"], pal), vmin=-vmax, vmax=vmax, aspect="auto"
    )
    probs_cmap = _cmap_with_masked(pal["attn_cmap"], pal)
    probs_plot = ax_probs.imshow(probs_image, cmap=probs_cmap, vmin=0, vmax=1, aspect="auto")

    if position < count - 1:
        # Grey out the future keys in the two rows that still show values, and mark the boundary.
        ax_scores.add_patch(Rectangle(
            (position + 0.5, -0.5), count - position - 1, 2,
            facecolor=pal["masked"], edgecolor="none", alpha=0.55, zorder=2,
        ))
        for axis in (ax_scores, ax_probs):
            axis.axvline(position + 0.5, color=pal["highlight"], linestyle="--", linewidth=1.2, zorder=4)

    if annotate:
        fontsize = float(np.clip(150.0 / count, 5.0, 8.0))
        for row in range(3):
            for col in range(count):
                value = scores[row, col]
                if np.isnan(value):
                    ax_scores.text(col, row, "−∞", ha="center", va="center",
                                   fontsize=fontsize, color=pal["text"], alpha=0.8)
                    continue
                strong = abs(value) / vmax > 0.6
                ax_scores.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=fontsize,
                               color=pal["on_strong"] if strong else pal["on_weak"])
        for col in range(count):
            value = probs_image[0, col]
            if np.isnan(value):
                ax_probs.text(col, 0, "0", ha="center", va="center",
                              fontsize=fontsize, color=pal["text"], alpha=0.8)
            else:
                ax_probs.text(col, 0, f"{value:.2f}", ha="center", va="center", fontsize=fontsize,
                              color="white" if value < 0.5 else "black")

    labels = [_tick_label(token) for token in tokens]
    tick_size = _token_fontsize(count)
    ax_probs.set_xticks(range(count))
    ax_probs.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_size, fontfamily="monospace")
    ax_probs.get_xticklabels()[position].set_color(pal["highlight"])
    ax_probs.get_xticklabels()[position].set_fontweight("bold")
    ax_scores.tick_params(labelbottom=False)
    ax_scores.set_yticks([0, 1, 2])
    ax_scores.set_yticklabels(
        [r"raw  $QK^{\top}$", r"scaled  $/\sqrt{d_k}$", "causal mask"], fontsize=8
    )
    ax_probs.set_yticks([0])
    ax_probs.set_yticklabels(["softmax"], fontsize=8)
    ax_probs.set_xlabel("Key token")

    _style_colorbar(fig.colorbar(scores_image, cax=cax_scores), "score", pal, labelsize=7)
    _style_colorbar(fig.colorbar(probs_plot, cax=cax_probs), "prob.", pal, labelsize=7)

    return fig


# --------------------------------------------------------------------------- #
# Hidden state of one token across the residual stream
# --------------------------------------------------------------------------- #

def plot_hidden_states(labels, matrix, theme="light", figsize=(7, 2.8)):
    """
    One token's hidden vector at every residual-stream stage.

    ``matrix`` is [stages, d_model]; ``labels`` names the stages (top to bottom). The heatmap
    scales every row by its own largest |value| so the *pattern* of each stage is visible
    whatever its magnitude; the bars on the right give the true L2 norm of each stage.
    Dimensions are plotted by index only; no meaning is attached to any of them.
    """
    pal = _palette(theme)
    data = _to_numpy(matrix).astype(float)
    stages, dim = data.shape
    row_max = np.abs(data).max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    norms = np.linalg.norm(data, axis=1)

    fig, (ax_heat, ax_norm) = plt.subplots(
        1, 2, sharey=True, figsize=figsize, facecolor=pal["fig_bg"],
        gridspec_kw={"width_ratios": [4, 1.3]}, layout="constrained",
    )
    _style_axes(ax_heat, pal)
    _style_axes(ax_norm, pal)

    image = ax_heat.imshow(data / row_max, cmap=pal["div_cmap"], vmin=-1, vmax=1, aspect="auto")
    ax_heat.set_yticks(range(stages))
    ax_heat.set_yticklabels(labels, fontsize=8)
    ax_heat.set_xlabel(f"Hidden dimension (0–{dim - 1})")

    ax_norm.barh(range(stages), norms, color=pal["bar_top"], height=0.7)
    for row, value in enumerate(norms):
        ax_norm.text(value, row, f" {value:.2f}", va="center", ha="left", fontsize=7, color=pal["text"])
    ax_norm.set_xlim(0, float(norms.max()) * 1.35 or 1.0)
    ax_norm.set_xlabel("L2 norm")
    ax_norm.tick_params(labelsize=7)
    ax_heat.set_ylim(stages - 0.5, -0.5)

    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.03, pad=0.02)
    _style_colorbar(cbar, "value / row max |v|", pal, labelsize=7)

    return fig