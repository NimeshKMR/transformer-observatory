"""
Transformer Observatory — compact inspection workbench for a GPT-style Transformer.

Layout
    header ........ branding, model status, model metadata
    prompt row .... prompt + Inspect button
    control row ... token position / layer / head + selected-token readout
    2x2 grid ...... attention map      | next-token predictions
                    Q/K/V inspector    | token inspector
    details ....... tabs: attention scores (pre-softmax stages), hidden states,
                    diagnostics, exact Q/K/V values, full-precision probabilities

Sections: configuration, model loading, analysis helpers (no UI), session-state
actions, rendering, main. The numerical work lives in transformer/analysis.py.
"""

import html
import inspect

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from transformer.analysis import (
    activation_summary,
    all_finite,
    attention_checks,
    attention_summary,
    describe_stage,
    finite_report,
    head_attention_stages,
    hidden_state_matrix,
    hidden_state_trajectory,
    layer_norm_input_stats,
    query_row,
    short_stage_label,
)
from transformer.bpe import BPETokeniser
from transformer.model import GPT
from transformer.observatory import inspect_model, next_token_distribution
from transformer.visualization import (
    plot_attention,
    plot_attention_stages,
    plot_hidden_states,
    plot_predictions,
    plot_qkv,
)


# =========================================================================== #
# 1. Configuration
# =========================================================================== #

CHECKPOINT_PATH = "checkpoints/best_gpt_bpe.pt"
TOKENISER_PATH = "checkpoints/bpe_tokeniser.json"

DEFAULT_PROMPT = "First Citizen:\n"
TOP_K = 10
CONTEXT_RADIUS = 3  # tokens shown either side of the selected token

PLOT_THEME = "dark"
ATTENTION_FIGSIZE = (6.6, 3.8)
PREDICTION_FIGSIZE = (4.4, 3.6)
QKV_FIGSIZE = (6.6, 2.1)
STAGES_FIGSIZE = (7.4, 3.3)
HIDDEN_FIGSIZE = (7.4, 2.8)

CHECK_TOLERANCE = 1e-5  # float32 numerical tolerance for the structural attention checks

ROW_HIGHLIGHT = "background-color: rgba(45, 212, 191, 0.22)"

# Newer Streamlit stretches dataframes by default and deprecates use_container_width;
# older versions need it passed explicitly. Detect which one we are running on.
STRETCH = (
    {}
    if inspect.signature(st.dataframe).parameters.get("width") is not None
    and inspect.signature(st.dataframe).parameters["width"].default == "stretch"
    else {"use_container_width": True}
)

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Transformer Observatory",
    page_icon="🔬",
    layout="wide",
)

STYLE = """
<style>
[data-testid="stMainBlockContainer"], .block-container { padding: 2.4rem 1.25rem 1.5rem; }
[data-testid="stVerticalBlock"] { gap: 0.65rem; }

.obs-header { display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 0.5rem 1.25rem; padding: 0.45rem 0.75rem;
  border: 1px solid rgba(128, 148, 176, 0.28); border-radius: 6px; }
.obs-brand { font-weight: 650; font-size: 1.05rem; letter-spacing: 0.02em; }
.obs-brand small { font-weight: 400; font-size: 0.8rem; opacity: 0.65;
  margin-left: 0.6rem; letter-spacing: 0; }
.obs-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; }
.obs-meta { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.72rem; padding: 0.12rem 0.45rem; white-space: nowrap;
  border: 1px solid rgba(128, 148, 176, 0.28); border-radius: 4px; }
.obs-meta b { color: #2dd4bf; font-weight: 600; }
.obs-status { color: #2dd4bf; }

.obs-panel-title { display: flex; justify-content: space-between; gap: 0.75rem;
  font-size: 0.72rem; font-weight: 650; letter-spacing: 0.09em;
  text-transform: uppercase; color: #2dd4bf; }
.obs-panel-title span { font-weight: 400; letter-spacing: 0; text-transform: none;
  opacity: 0.75; color: inherit;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }

.obs-label { font-size: 0.875rem; opacity: 0.7; margin-bottom: 0.25rem; }
.obs-selected { display: flex; align-items: center; gap: 0.6rem; min-height: 2.5rem;
  padding: 0 0.75rem; overflow: hidden; white-space: nowrap;
  border: 1px solid rgba(128, 148, 176, 0.28); border-radius: 0.5rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.85rem; }
.obs-selected .obs-pos { color: #2dd4bf; }
.obs-selected .obs-id { opacity: 0.6; margin-left: auto; }
.obs-selected .obs-tok { overflow: hidden; text-overflow: ellipsis; }

.obs-kv > div { display: flex; justify-content: space-between; gap: 1rem;
  padding: 0.2rem 0; font-size: 0.82rem;
  border-bottom: 1px solid rgba(128, 148, 176, 0.18); }
.obs-kv span { opacity: 0.65; }
.obs-kv b { font-weight: 500; text-align: right;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }

.obs-ctx { display: flex; flex-wrap: wrap; align-items: stretch; gap: 0.3rem; }
.obs-chip { display: flex; flex-direction: column; padding: 0.15rem 0.45rem;
  border: 1px solid rgba(128, 148, 176, 0.28); border-radius: 4px; line-height: 1.25;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.78rem; }
.obs-chip b { font-weight: 500; }
.obs-chip i { font-style: normal; font-size: 0.62rem; opacity: 0.6; }
.obs-chip.sel { border-color: #2dd4bf; background: rgba(45, 212, 191, 0.12); }
.obs-gap { align-self: center; opacity: 0.5; }
</style>
"""


# =========================================================================== #
# 2. Model loading (cached)
# =========================================================================== #

@st.cache_resource
def load_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    tokenizer = BPETokeniser.load(TOKENISER_PATH)

    model = GPT(
        vocab_size=checkpoint["vocab_size"],
        context_length=checkpoint["context_length"],
        embedding_dim=checkpoint["embedding_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, checkpoint


# =========================================================================== #
# 3. Analysis helpers (no UI)
# =========================================================================== #

def validate_prompt(prompt, tokenizer, context_length):
    """Return the token ids for `prompt`, or raise ValueError with a clear message."""
    token_ids = tokenizer.encode(prompt)

    if not token_ids:
        raise ValueError("Prompt must contain at least one token.")

    if len(token_ids) > context_length:
        raise ValueError(
            f"Prompt has {len(token_ids)} tokens, but the model "
            f"supports a maximum context length of {context_length}."
        )

    return token_ids


def get_predictions(result, tokenizer, position):
    """
    Top-k predictions for `position`.

    One forward pass over the prompt already produced the logits of every position
    (`result["logits"]`, shape [T, vocab]), so changing the selected position only
    needs a softmax and top-k on one row. Changing position, layer or head never
    runs the model again.
    """
    _, predictions = next_token_distribution(result["logits"][position], tokenizer, k=TOP_K)
    return predictions


def get_diagnostics(result):
    """Diagnostics for the current inspection, computed once and kept in session state."""
    prompt = st.session_state["inspection_prompt"]
    cached = st.session_state.get("diagnostics")

    if cached is None or cached["prompt"] != prompt:
        hidden = result["hidden_states"]
        cached = {
            "prompt": prompt,
            "finite": finite_report(result),
            "checks": attention_checks(result, tolerance=CHECK_TOLERANCE),
            "attention": attention_summary(result),
            "activations": activation_summary(hidden),
            "layernorm": layer_norm_input_stats(hidden),
        }
        st.session_state["diagnostics"] = cached

    return cached


def get_qkv(result, layer_idx, head_idx, position):
    """Q, K, V vectors (each of length head_dim) for one token, layer and head."""

    def vector(name):
        # result[name][layer] has shape [num_heads, seq_len, head_dim]
        raw = result[name][layer_idx][head_idx, position]
        return torch.as_tensor(raw).detach().cpu().float().numpy()

    return {
        "Query (Q)": vector("queries"),
        "Key (K)": vector("keys"),
        "Value (V)": vector("values"),
    }


# =========================================================================== #
# 4. Session-state actions
# =========================================================================== #

def run_inspection(model, tokenizer):
    """
    Button callback. Runs before the script reruns, so it may safely set the
    keys of widgets (such as the position selector) that are created later.
    """
    prompt = st.session_state["prompt_text"]
    st.session_state["inspect_error"] = None

    try:
        token_ids = validate_prompt(prompt, tokenizer, model.context_length)
        position = len(token_ids) - 1
        result = inspect_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            position=position,
        )
    except ValueError as error:
        message = str(error)
    except KeyError as error:
        message = f"Tokenisation or lookup failed: missing key {error}."
    except Exception as error:  # surface anything else in the UI, not as a traceback
        message = f"Inspection failed ({type(error).__name__}): {error}"
    else:
        st.session_state["inspection_result"] = result
        st.session_state["inspection_prompt"] = prompt
        st.session_state["prediction_position"] = position
        st.session_state.pop("diagnostics", None)
        return

    st.session_state["inspect_error"] = (prompt, message)


# =========================================================================== #
# 5. Rendering
# =========================================================================== #

def esc(value):
    """Escape text for inclusion in HTML passed to st.markdown."""
    return html.escape(str(value)).replace("$", "&#36;")


def show_figure(fig):
    try:
        st.pyplot(fig)
    finally:
        plt.close(fig)


def panel_header(title, meta=""):
    st.markdown(
        f'<div class="obs-panel-title">{esc(title)}<span>{esc(meta)}</span></div>',
        unsafe_allow_html=True,
    )


def kv_html(rows):
    body = "".join(f"<div><span>{esc(k)}</span><b>{esc(v)}</b></div>" for k, v in rows)
    return f'<div class="obs-kv">{body}</div>'


def context_html(tokens, token_ids, position):
    lo = max(0, position - CONTEXT_RADIUS)
    hi = min(len(tokens), position + CONTEXT_RADIUS + 1)

    parts = []
    if lo > 0:
        parts.append('<span class="obs-gap">…</span>')
    for i in range(lo, hi):
        cls = "obs-chip sel" if i == position else "obs-chip"
        parts.append(
            f'<span class="{cls}"><b>{esc(repr(tokens[i]))}</b>'
            f"<i>{i} · #{token_ids[i]}</i></span>"
        )
    if hi < len(tokens):
        parts.append('<span class="obs-gap">…</span>')

    return f'<div class="obs-ctx">{"".join(parts)}</div>'


def render_header(tokenizer, checkpoint):
    meta = [
        ("device", str(device)),
        ("vocab", f"{tokenizer.vocab_size:,}"),
        ("layers", checkpoint["num_layers"]),
        ("heads", checkpoint["num_heads"]),
        ("d_model", checkpoint["embedding_dim"]),
        ("d_head", checkpoint["embedding_dim"] // checkpoint["num_heads"]),
        ("ctx", checkpoint["context_length"]),
    ]
    chips = "".join(
        f'<span class="obs-meta">{esc(key)} <b>{esc(value)}</b></span>'
        for key, value in meta
    )
    st.markdown(
        '<div class="obs-header">'
        '<div class="obs-brand">Transformer Observatory'
        "<small>GPT internal-state inspection</small></div>"
        '<div class="obs-chips">'
        '<span class="obs-meta"><span class="obs-status">●</span> model ready · eval</span>'
        f"{chips}</div></div>",
        unsafe_allow_html=True,
    )


def render_prompt_row(model, tokenizer):
    prompt_col, button_col = st.columns([8, 1], gap="small")

    with prompt_col:
        prompt = st.text_area(
            "Prompt",
            key="prompt_text",
            height=68,
            label_visibility="collapsed",
            placeholder="Enter a prompt to inspect…",
        )

    with button_col:
        st.button(
            "Inspect",
            type="primary",
            on_click=run_inspection,
            args=(model, tokenizer),
        )

    return prompt


def render_controls(result, checkpoint):
    tokens = result["tokens"]
    token_ids = result["token_ids"]

    col_position, col_layer, col_head, col_selected = st.columns([3, 1, 1, 3], gap="small")

    with col_position:
        position = st.selectbox(
            "Token position",
            options=list(range(len(tokens))),
            format_func=lambda i: f"{i}: {tokens[i]!r}",
            key="prediction_position",
        )

    with col_layer:
        layer_idx = st.selectbox(
            "Layer", range(checkpoint["num_layers"]), key="layer_idx"
        )

    with col_head:
        head_idx = st.selectbox(
            "Head", range(checkpoint["num_heads"]), key="head_idx"
        )

    with col_selected:
        st.markdown(
            '<div class="obs-label">Selected token</div>'
            '<div class="obs-selected">'
            f'<span class="obs-pos">pos {position}</span>'
            f'<span class="obs-tok">{esc(repr(tokens[position]))}</span>'
            f'<span class="obs-id">id {esc(token_ids[position])}</span>'
            "</div>",
            unsafe_allow_html=True,
        )

    return position, layer_idx, head_idx


def render_attention_panel(result, layer_idx, head_idx, position):
    with st.container(border=True):
        panel_header("Attention map", f"L{layer_idx} · H{head_idx} · query pos {position}")
        show_figure(
            plot_attention(
                result,
                layer_idx=layer_idx,
                head_idx=head_idx,
                position=position,
                theme=PLOT_THEME,
                figsize=ATTENTION_FIGSIZE,
                show_title=False,
            )
        )
        st.caption(
            "Rows are query tokens, columns are the keys they attend to; the highlighted "
            "row is the selected position. Each row sums to 1 over the keys it can see "
            "(causal mask). `·` = space, `\\n` = newline."
        )


def render_predictions_panel(view, position):
    predictions = view["top_predictions"]
    token = view["tokens"][position]

    with st.container(border=True):
        panel_header("Next-token predictions", f"after pos {position} {token!r}")

        chart_col, table_col = st.columns([5, 4], gap="small")

        with chart_col:
            show_figure(
                plot_predictions(
                    view,
                    top_k=TOP_K,
                    theme=PLOT_THEME,
                    figsize=PREDICTION_FIGSIZE,
                    show_title=False,
                )
            )

        with table_col:
            table = pd.DataFrame(
                [
                    {
                        "Token": repr(item["token"]),
                        "Token ID": item["token_id"],
                        "Probability": item["probability"],
                    }
                    for item in predictions
                ]
            )
            st.dataframe(
                table,
                **STRETCH,
                hide_index=True,
                height=min(len(table), TOP_K) * 35 + 38,
                column_config={
                    "Probability": st.column_config.NumberColumn(format="%.4f"),
                },
            )

        st.caption("Chart labels: `·` = space, `\\n` = newline. The table shows each token's exact `repr`.")


def render_qkv_panel(qkv, layer_idx, head_idx, position, token):
    dim = len(next(iter(qkv.values())))
    norms = " · ".join(
        f"‖{name[0]}‖ {float((vec @ vec) ** 0.5):.3f}" for name, vec in qkv.items()
    )

    with st.container(border=True):
        panel_header("Q / K / V inspector", f"L{layer_idx} · H{head_idx} · pos {position} {token!r}")
        show_figure(plot_qkv(qkv, theme=PLOT_THEME, figsize=QKV_FIGSIZE))
        st.caption(
            f"head_dim = {dim} · {norms} · one colour scale shared by all rows "
            "(teal positive, amber negative). Exact values are under Details."
        )


def render_token_panel(result, position):
    tokens = result["tokens"]
    token_ids = result["token_ids"]
    count = len(tokens)

    with st.container(border=True):
        panel_header("Token inspector", f"pos {position} of {count - 1}")

        tab_selected, tab_all = st.tabs(["Selected token", "Full tokenization"])

        with tab_selected:
            st.markdown(
                kv_html(
                    [
                        ("Token", repr(tokens[position])),
                        ("Token ID", token_ids[position]),
                        ("Position", position),
                        ("Visible prefix", f"{position + 1} token(s), causal mask"),
                        ("Prompt length", f"{count} tokens"),
                    ]
                ),
                unsafe_allow_html=True,
            )
            st.caption("Neighbouring tokens (position · token ID)")
            st.markdown(context_html(tokens, token_ids, position), unsafe_allow_html=True)

        with tab_all:
            frame = pd.DataFrame(
                {
                    "Position": list(range(count)),
                    "Token ID": token_ids,
                    "Token": [repr(token) for token in tokens],
                }
            )
            styled = frame.style.apply(
                lambda row: [ROW_HIGHLIGHT if row["Position"] == position else "" for _ in row],
                axis=1,
            )
            st.dataframe(styled, **STRETCH, hide_index=True, height=240)


def render_attention_scores_tab(result, layer_idx, head_idx, position):
    """Pre-softmax attention calculation for the selected query, recomputed from the stored Q and K."""
    tokens = result["tokens"]
    count = len(tokens)
    head_dim = result["queries"][layer_idx].shape[-1]

    stages = head_attention_stages(result, layer_idx, head_idx)
    row = query_row(stages, position)
    stored = result["attention_weights"][layer_idx][head_idx]
    difference = float((stages["probabilities"] - stored).abs().max())

    st.caption(
        f"Query pos {position} {tokens[position]!r} · layer {layer_idx} · head {head_idx} · "
        f"d_k = {head_dim}. Stages: raw = q·kᵀ → scaled = raw / √d_k → causal mask "
        f"(future keys → −∞) → softmax. Recomputed from the stored Q and K; "
        f"max |difference| from the forward-pass weights: {difference:.1e}."
    )
    show_figure(
        plot_attention_stages(
            row["raw"],
            row["scaled"],
            row["masked"],
            row["probabilities"],
            tokens,
            position,
            theme=PLOT_THEME,
            figsize=STAGES_FIGSIZE,
        )
    )

    visible = row["visible"].tolist()
    frame = pd.DataFrame(
        {
            "Key pos": list(range(count)),
            "Key token": [repr(token) for token in tokens],
            "Visible": visible,
            "Raw": row["raw"].tolist(),
            "Scaled": row["scaled"].tolist(),
            "After mask": row["masked"].tolist(),
            "Probability": row["probabilities"].tolist(),
        }
    )
    shown = frame.copy()
    shown["After mask"] = [
        f"{value:.6f}" if ok else "−∞" for ok, value in zip(visible, frame["After mask"])
    ]

    with st.expander("Numerical values for this query"):
        st.dataframe(
            shown,
            **STRETCH,
            hide_index=True,
            height=min(count, 8) * 35 + 38,
            column_config={
                "Raw": st.column_config.NumberColumn(format="%.6f"),
                "Scaled": st.column_config.NumberColumn(format="%.6f"),
                "Probability": st.column_config.NumberColumn(format="%.6f"),
            },
        )
        st.download_button(
            "Download CSV",
            frame.to_csv(index=False),
            file_name=f"attention_scores_layer{layer_idx}_head{head_idx}_query{position}.csv",
            mime="text/csv",
        )


def render_hidden_states_tab(result, position):
    """The selected token's vector at every residual-stream stage."""
    hidden = result["hidden_states"]
    token = result["tokens"][position]

    labels, matrix = hidden_state_matrix(hidden, position)
    trajectory = hidden_state_trajectory(hidden, position)

    st.caption(
        f"Token pos {position} {token!r}. Each block adds an attention update and an MLP "
        "update to the residual stream: after attention = X + attn(LN1(X)); block output = "
        "that + MLP(LN2(·)). The final LayerNorm output is what the LM head reads, so its norm "
        "is not comparable with the stages before it. Dimensions are shown by index only; "
        "nothing here says what any dimension encodes."
    )
    show_figure(
        plot_hidden_states(
            [short_stage_label(label) for label in labels],
            matrix,
            theme=PLOT_THEME,
            figsize=HIDDEN_FIGSIZE,
        )
    )

    with st.expander("Numerical values"):
        table = pd.DataFrame(
            [
                {
                    "Stage": row["name"],
                    "Shape": str(row["shape"]),
                    "L2 norm": row["l2_norm"],
                    "Mean": row["mean"],
                    "Std": row["std"],
                    "Min": row["min"],
                    "Max": row["max"],
                    "Cosine to previous": row["cosine_to_previous"],
                    "Relative change": row["relative_change"],
                }
                for row in trajectory
            ]
        )
        st.dataframe(
            table,
            **STRETCH,
            hide_index=True,
            column_config={
                name: st.column_config.NumberColumn(format="%.4f")
                for name in table.columns
                if name not in ("Stage", "Shape")
            },
        )
        st.caption(
            "Mean/Std/Min/Max are over the hidden dimensions of this token (population std). "
            "Relative change = ‖hₖ − hₖ₋₁‖ / ‖hₖ₋₁‖; cosine and relative change describe geometry only."
        )

        vectors = pd.DataFrame(
            matrix.numpy(),
            columns=[f"dim_{i}" for i in range(matrix.shape[1])],
        )
        vectors.insert(0, "Stage", [describe_stage(label) for label in labels])
        st.dataframe(
            vectors,
            **STRETCH,
            hide_index=True,
            column_config={
                name: st.column_config.NumberColumn(format="%.5f")
                for name in vectors.columns
                if name != "Stage"
            },
        )
        st.download_button(
            "Download vectors (CSV)",
            vectors.to_csv(index=False),
            file_name=f"hidden_states_pos{position}.csv",
            mime="text/csv",
        )


def render_diagnostics_tab(result):
    """Measurements from this forward pass (independent of the selected position, layer and head)."""
    diagnostics = get_diagnostics(result)

    nonfinite = sum(row["nan"] + row["inf"] for row in diagnostics["finite"])
    checks = diagnostics["checks"]
    failing = [row for row in checks if not row["ok"]]
    worst_row_sum = max(row["max_row_sum_error"] for row in checks)
    worst_future = max(row["max_future_weight"] for row in checks)
    worst_recon = max(row["max_reconstruction_error"] for row in checks)

    st.markdown(
        kv_html(
            [
                ("NaN / Inf values in stored tensors", "none found" if all_finite(diagnostics["finite"]) else f"{nonfinite} found"),
                ("Attention row sums: max |Σ − 1|", f"{worst_row_sum:.2e}"),
                ("Attention to future keys: max weight", f"{worst_future:.2e}"),
                ("Recomputed vs forward-pass weights: max |diff|", f"{worst_recon:.2e}"),
                (f"Heads passing all three checks (tolerance {CHECK_TOLERANCE:g})", f"{len(checks) - len(failing)} of {len(checks)}"),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "Measurements of this one forward pass. The tolerance is a float32 numerical tolerance "
        "chosen for this tool, not a published standard; passing these structural checks shows the "
        "masking and instrumentation behave as designed, not that the model is healthy."
    )

    column_formats = lambda frame, skip: {  # noqa: E731 - tiny local helper
        name: st.column_config.NumberColumn(format="%.4g")
        for name in frame.columns
        if name not in skip
    }

    with st.expander("Activations by stage (all tokens)", expanded=True):
        frame = pd.DataFrame(diagnostics["activations"]).drop(columns=["stage"])
        frame["shape"] = frame["shape"].astype(str)
        frame = frame.rename(columns={"name": "Stage"})
        st.dataframe(frame, **STRETCH, hide_index=True, column_config=column_formats(frame, ("Stage", "shape")))

    with st.expander("LayerNorm inputs (per-token statistics over the feature dimension)"):
        frame = pd.DataFrame(diagnostics["layernorm"])
        st.dataframe(frame, **STRETCH, hide_index=True, column_config=column_formats(frame, ("layernorm", "input_stage")))

    with st.expander("Attention structure checks (per head)"):
        frame = pd.DataFrame(checks)
        st.dataframe(frame, **STRETCH, hide_index=True, column_config=column_formats(frame, ("layer", "head", "ok")))

    with st.expander("Attention distribution (descriptive, this prompt only)"):
        frame = pd.DataFrame(diagnostics["attention"])
        st.dataframe(frame, **STRETCH, hide_index=True, column_config=column_formats(frame, ("layer", "head")))
        st.caption(
            "Entropy in nats; normalized entropy 1.0 means uniform over the keys a query may see. "
            "These numbers describe where attention mass goes for this prompt; they do not assign "
            "a function to any head."
        )

    with st.expander("Non-finite check per tensor"):
        frame = pd.DataFrame(diagnostics["finite"])
        frame["shape"] = frame["shape"].astype(str)
        st.dataframe(frame, **STRETCH, hide_index=True)


def render_detail_tabs(result, qkv, predictions, layer_idx, head_idx, position):
    st.markdown('<div class="obs-panel-title">Details</div>', unsafe_allow_html=True)
    tab_scores, tab_hidden, tab_diagnostics, tab_qkv, tab_probabilities = st.tabs(
        ["Attention scores", "Hidden states", "Diagnostics", "Q/K/V values", "Probabilities"]
    )

    with tab_scores:
        render_attention_scores_tab(result, layer_idx, head_idx, position)

    with tab_hidden:
        render_hidden_states_tab(result, position)

    with tab_diagnostics:
        render_diagnostics_tab(result)

    with tab_qkv:
        dim = len(next(iter(qkv.values())))
        frame = pd.DataFrame(
            [
                {"Vector": name, **{f"dim_{i}": float(x) for i, x in enumerate(vec)}}
                for name, vec in qkv.items()
            ]
        )
        st.caption(
            f"Layer {layer_idx} · head {head_idx} · position {position}. "
            "The table shows 6 decimals; the CSV keeps full precision."
        )
        st.dataframe(
            frame,
            **STRETCH,
            hide_index=True,
            column_config={
                f"dim_{i}": st.column_config.NumberColumn(format="%.6f")
                for i in range(dim)
            },
        )
        st.download_button(
            "Download CSV",
            frame.to_csv(index=False),
            file_name=f"qkv_layer{layer_idx}_head{head_idx}_pos{position}.csv",
            mime="text/csv",
        )

    with tab_probabilities:
        frame = pd.DataFrame(
            [
                {
                    "Rank": rank,
                    "Token": repr(item["token"]),
                    "Token ID": item["token_id"],
                    "Probability": item["probability"],
                    "Logit": float(result["logits"][position, item["token_id"]]),
                }
                for rank, item in enumerate(predictions, start=1)
            ]
        )
        st.caption(
            f"Full-precision probabilities for the prediction after position {position}, "
            "with the raw logit (pre-softmax output of the LM head) of each token."
        )
        st.dataframe(
            frame,
            **STRETCH,
            hide_index=True,
            column_config={
                "Probability": st.column_config.NumberColumn(format="%.8f"),
                "Logit": st.column_config.NumberColumn(format="%.6f"),
            },
        )


# =========================================================================== #
# 6. Main
# =========================================================================== #

def main():
    try:
        model, tokenizer, checkpoint = load_model()
    except FileNotFoundError as error:
        st.error(
            f"Could not load model assets: {error}. Expected `{CHECKPOINT_PATH}` and "
            f"`{TOKENISER_PATH}` relative to the directory Streamlit is launched from."
        )
        st.stop()

    st.markdown(STYLE, unsafe_allow_html=True)
    render_header(tokenizer, checkpoint)

    st.session_state.setdefault("prompt_text", DEFAULT_PROMPT)
    prompt = render_prompt_row(model, tokenizer)

    error = st.session_state.get("inspect_error")
    has_error = bool(error) and error[0] == prompt
    if has_error:
        st.error(error[1])

    result = st.session_state.get("inspection_result")
    if result is None or st.session_state.get("inspection_prompt") != prompt:
        if not has_error:
            if result is None:
                st.info("Enter a prompt and press **Inspect** to open the workbench.")
            else:
                st.info("The prompt changed since the last inspection. Press **Inspect** to analyse it.")
        return

    position, layer_idx, head_idx = render_controls(result, checkpoint)

    try:
        predictions = get_predictions(result, tokenizer, position)
    except Exception as exc:
        st.error(f"Could not compute predictions for position {position}: {exc}")
        return

    view = {**result, "top_predictions": predictions}
    qkv = get_qkv(result, layer_idx, head_idx, position)
    token = result["tokens"][position]

    top_left, top_right = st.columns(2, gap="small")
    bottom_left, bottom_right = st.columns(2, gap="small")

    with top_left:
        render_attention_panel(result, layer_idx, head_idx, position)
    with top_right:
        render_predictions_panel(view, position)
    with bottom_left:
        render_qkv_panel(qkv, layer_idx, head_idx, position, token)
    with bottom_right:
        render_token_panel(result, position)

    render_detail_tabs(result, qkv, predictions, layer_idx, head_idx, position)


main()