import io
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_COLORS = ['#00cc44', '#4488ff', '#ffdd00', '#ff4444']


def parse_profile_file(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """
    Parse a wide-format profile .txt file.
    Returns DataFrame: rows = positions (0-based), columns = event IDs (as strings).
    """
    try:
        text = file_bytes.decode('utf-8')
    except UnicodeDecodeError:
        text = file_bytes.decode('latin-1')

    lines = text.splitlines()
    if not lines:
        raise ValueError(f"Profile file '{file_name}' is empty.")

    header_line = lines[0]
    sep = '\t' if '\t' in header_line else ','
    n_cols = len(header_line.split(sep))

    if n_cols < 2:
        raise ValueError(
            f"Profile file '{file_name}' does not appear to be a valid profile export "
            f"(only {n_cols} column found in header)."
        )

    # Keep only lines with the correct number of fields (skips footer text)
    kept = [header_line]
    for line in lines[1:]:
        if len(line.split(sep)) == n_cols:
            kept.append(line)

    if len(kept) < 2:
        raise ValueError(f"No data rows found in profile file '{file_name}'.")

    df = pd.read_csv(
        io.StringIO('\n'.join(kept)),
        sep=sep,
        low_memory=False,
    )
    # Drop any trailing empty column caused by a trailing separator
    df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]
    df = df.dropna(axis=1, how='all')
    # Convert to int32 after dropping bad columns; coerce any remaining NaN to 0
    df = df.fillna(0).astype(np.int32)
    return df


def infer_channel_label(file_name: str) -> str:
    """Guess a short channel label from the filename (e.g. 'profiles_ch2_prf.txt' → 'ch2')."""
    stem = file_name.rsplit('.', 1)[0]
    m = re.search(r'ch\d+', stem, re.IGNORECASE)
    if m:
        return m.group(0)
    return stem[:20]


def build_profile_matrix(
    profile_df: pd.DataFrame,
    event_ids: list,
    tof_series: pd.Series,
    max_events: int = 500,
) -> tuple:
    """
    Build a 2D matrix for the heatmap and overlay.

    Returns
    -------
    matrix       : np.ndarray shape (n_events, n_positions), float32
    sorted_ids   : list of int event IDs (sorted by TOF descending — largest first)
    sorted_tofs  : list of int TOF values corresponding to sorted_ids
    """
    available = set(profile_df.columns.astype(str))
    matched = [eid for eid in event_ids if str(eid) in available]

    if not matched:
        return np.empty((0, 0), dtype=np.float32), [], []

    # Sort descending: Plotly draws row-0 at the bottom, so largest TOF lands at
    # the bottom and smallest TOF at the top — matching the requested layout.
    tof_for_matched = {eid: int(tof_series.get(eid, 0)) for eid in matched}
    sorted_ids = sorted(matched, key=lambda e: tof_for_matched[e], reverse=True)
    sorted_tofs = [tof_for_matched[e] for e in sorted_ids]

    # Evenly subsample if too many events (preserves size distribution)
    if len(sorted_ids) > max_events:
        indices = np.round(np.linspace(0, len(sorted_ids) - 1, max_events)).astype(int)
        sorted_ids  = [sorted_ids[i]  for i in indices]
        sorted_tofs = [sorted_tofs[i] for i in indices]

    cols = [str(eid) for eid in sorted_ids]
    matrix = profile_df[cols].to_numpy(dtype=np.float32).T  # shape: (n_events, n_positions)
    return matrix, sorted_ids, sorted_tofs


def build_aligned_matrices(
    profile_dfs: dict,
    event_ids: list,
    tof_series: pd.Series,
    max_events: int = 500,
) -> tuple:
    """
    Build matrices for multiple channels with identical event sets and ordering.
    Only events present in ALL channel DataFrames are included.

    Parameters
    ----------
    profile_dfs : dict  label → DataFrame (from parse_profile_file)
    event_ids   : list of int event IDs for the selected well
    tof_series  : pd.Series  event_id → TOF
    max_events  : cap on rows (evenly sampled)

    Returns
    -------
    matrices    : dict  label → np.ndarray (n_events, n_positions)
    sorted_ids  : list of int
    sorted_tofs : list of int
    """
    # Intersection of event IDs across all channel files
    common = set(str(eid) for eid in event_ids)
    for df in profile_dfs.values():
        common &= set(df.columns.astype(str))
    matched = [eid for eid in event_ids if str(eid) in common]

    if not matched:
        return {}, [], []

    tof_for_matched = {eid: int(tof_series.get(eid, 0)) for eid in matched}
    sorted_ids = sorted(matched, key=lambda e: tof_for_matched[e], reverse=True)
    sorted_tofs = [tof_for_matched[e] for e in sorted_ids]

    if len(sorted_ids) > max_events:
        indices = np.round(np.linspace(0, len(sorted_ids) - 1, max_events)).astype(int)
        sorted_ids  = [sorted_ids[i]  for i in indices]
        sorted_tofs = [sorted_tofs[i] for i in indices]

    cols = [str(eid) for eid in sorted_ids]
    matrices = {label: df[cols].to_numpy(dtype=np.float32).T
                for label, df in profile_dfs.items()}
    return matrices, sorted_ids, sorted_tofs


# ── Color helpers ─────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> tuple:
    """'#rrggbb' → (r, g, b) integers 0-255."""
    h = hex_color.lstrip('#')
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)'."""
    r, g, b = _hex_to_rgb(hex_color)
    return f'rgba({r},{g},{b},{alpha})'


# ── Shared tick helper ────────────────────────────────────────────────────────

def _tof_ticks(sorted_tofs: list, n_events: int) -> tuple:
    tick_step = max(1, n_events // 20)
    tick_vals = list(range(0, n_events, tick_step))
    tick_text = [f'TOF {sorted_tofs[i]}' for i in tick_vals]
    return tick_vals, tick_text


# ── Single-channel heatmap ────────────────────────────────────────────────────

def make_profile_heatmap(
    matrix: np.ndarray,
    sorted_tofs: list,
    well: str,
    channel_name: str,
    vmin: float,
    vmax: float,
    channel_color: str = '#00cc44',
    bg_color: str = 'white',
) -> go.Figure:
    """
    Stacked worm profile heatmap: smallest worms at top, largest at bottom.
    channel_color sets the high-intensity end; bg_color sets the zero/background end.
    zsmooth=False ensures raw data is shown without interpolation.
    """
    n_events, n_pos = matrix.shape
    colorscale = [[0.0, bg_color], [1.0, channel_color]]
    tick_vals, tick_text = _tof_ticks(sorted_tofs, n_events)

    trace = go.Heatmap(
        z=matrix,
        colorscale=colorscale,
        zmin=vmin,
        zmax=vmax,
        zsmooth=False,
        colorbar=dict(
            title=dict(text=channel_name, side='right', font=dict(color='black', size=13)),
            tickfont=dict(color='black', size=12),
        ),
        hovertemplate='Position: %{x}<br>Event row: %{y}<br>Intensity: %{z:.0f}<extra></extra>',
    )

    fig = go.Figure(data=[trace])
    fig.update_layout(
        title=dict(
            text=f'{channel_name} — {well}  (n={n_events})',
            font=dict(size=18, color='black'),
        ),
        xaxis=dict(
            title='Position along worm (bins)',
            title_font=dict(size=15, color='black'),
            tickfont=dict(size=13, color='black'),
            showline=True, linewidth=2, linecolor='black',
            ticks='outside', tickcolor='black',
            range=[-0.5, n_pos - 0.5],
        ),
        yaxis=dict(
            title='Worm events (sorted by size ↑)',
            title_font=dict(size=15, color='black'),
            showticklabels=False,
            showline=True, linewidth=2, linecolor='black',
            ticks='',
            range=[-0.5, n_events - 0.5],
        ),
        height=800,
        margin=dict(t=60, b=80, l=80, r=30),
        plot_bgcolor=bg_color,
        paper_bgcolor='white',
        font=dict(color='black'),
    )
    return fig


# ── Multi-channel side-by-side heatmap ────────────────────────────────────────

def make_multi_heatmap(
    channel_entries: list,
    well: str,
    bg_color: str = 'white',
) -> go.Figure:
    """
    Side-by-side heatmaps, one panel per channel, sharing y-axis.

    channel_entries : list of dicts with keys:
        'label'      : str
        'color'      : str  hex
        'matrix'     : np.ndarray (n_events, n_positions)
        'vmin'       : float
        'vmax'       : float
        'sorted_tofs': list of int
    """
    n = len(channel_entries)
    if n == 0:
        return go.Figure()

    n_events = channel_entries[0]['matrix'].shape[0]
    tick_vals, tick_text = _tof_ticks(channel_entries[0]['sorted_tofs'], n_events)

    hs = 0.04  # horizontal spacing between subplots
    fig = make_subplots(
        rows=1, cols=n,
        shared_yaxes=True,
        horizontal_spacing=hs,
        subplot_titles=[e['label'] for e in channel_entries],
    )

    # Compute subplot domain right-edges for colorbar placement
    subplot_width = (1.0 - (n - 1) * hs) / n
    cb_len = 1.0 / n

    for i, entry in enumerate(channel_entries, start=1):
        colorscale = [[0.0, bg_color], [1.0, entry['color']]]
        cb_y = 1.0 - (2 * i - 1) / (2 * n)  # stacked colorbars on right

        trace = go.Heatmap(
            z=entry['matrix'],
            colorscale=colorscale,
            zmin=entry['vmin'],
            zmax=entry['vmax'],
            zsmooth=False,
            colorbar=dict(
                title=dict(text=entry['label'], side='right',
                           font=dict(color='black', size=11)),
                tickfont=dict(color='black', size=10),
                x=1.02,
                y=cb_y,
                yanchor='middle',
                len=cb_len,
                thickness=12,
            ),
            hovertemplate=(
                f'<b>{entry["label"]}</b><br>'
                'Position: %{x}<br>Event row: %{y}<br>Intensity: %{z:.0f}<extra></extra>'
            ),
        )
        fig.add_trace(trace, row=1, col=i)

        n_pos_i = entry['matrix'].shape[1]
        fig.update_xaxes(
            title_text='Position (bins)',
            title_font=dict(color='black', size=12),
            tickfont=dict(color='black', size=11),
            showline=True, linewidth=2, linecolor='black',
            ticks='outside', tickcolor='black',
            range=[-0.5, n_pos_i - 0.5],
            row=1, col=i,
        )

    # Y-axis: title only, no tick labels (shared across all panels)
    fig.update_yaxes(
        title_text='Worm events (sorted by size ↑)',
        title_font=dict(color='black', size=13),
        showticklabels=False,
        ticks='',
        showline=True, linewidth=2, linecolor='black',
        range=[-0.5, n_events - 0.5],
        row=1, col=1,
    )

    fig.update_layout(
        title=dict(
            text=f'Multi-channel heatmap — {well}  (n={n_events})',
            font=dict(size=18, color='black'),
        ),
        height=800,
        plot_bgcolor=bg_color,
        paper_bgcolor='white',
        font=dict(color='black'),
        margin=dict(t=80, b=80, l=80, r=100),
    )
    return fig


# ── Composite overlay heatmap (additive RGB blending) ────────────────────────

def make_overlay_heatmap(
    channel_entries: list,
    well: str,
    bg_color: str = 'black',
) -> go.Figure:
    """
    Additive RGB composite overlay heatmap (fluorescence-microscopy style).
    Each channel's intensity is additively blended using its assigned color.
    Looks most natural on black background.

    channel_entries : same format as make_multi_heatmap, requires 'sorted_tofs'.
    """
    if not channel_entries:
        return go.Figure()

    n_events, n_positions = channel_entries[0]['matrix'].shape
    rgb = np.zeros((n_events, n_positions, 3), dtype=np.float32)

    for entry in channel_entries:
        mat = entry['matrix'].astype(np.float32)
        span = entry['vmax'] - entry['vmin']
        if span <= 0:
            continue
        norm = np.clip((mat - entry['vmin']) / span, 0.0, 1.0)
        r_ch, g_ch, b_ch = _hex_to_rgb(entry['color'])
        rgb[:, :, 0] += norm * (r_ch / 255.0)
        rgb[:, :, 1] += norm * (g_ch / 255.0)
        rgb[:, :, 2] += norm * (b_ch / 255.0)

    img_uint8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    # go.Image puts row-0 at visual TOP by default.
    # Our row-0 = largest TOF (should be at visual bottom).
    # Flip the array so row-0 (largest) ends up at the visual bottom.
    img_uint8 = img_uint8[::-1]

    legend_html = '  '.join(
        f'<span style="color:{e["color"]}">■</span> {e["label"]}'
        for e in channel_entries
    )

    fig = go.Figure(go.Image(z=img_uint8))
    fig.update_layout(
        title=dict(
            text=f'Overlay heatmap — {well}  (n={n_events})',
            font=dict(size=18, color='black'),
        ),
        xaxis=dict(
            title='Position along worm (bins)',
            title_font=dict(size=15, color='black'),
            tickfont=dict(color='black', size=12),
            showline=True, linewidth=1, linecolor='black',
            ticks='outside', tickcolor='black',
            range=[-0.5, n_positions - 0.5],
        ),
        yaxis=dict(
            title='Worm events (sorted by size ↑)',
            title_font=dict(size=15, color='black'),
            showticklabels=False,
            ticks='',
            showline=True, linewidth=1, linecolor='black',
            # No range override: go.Image auto-reverses y (row-0 at top),
            # which correctly places the flipped array (row-0 = smallest TOF) at top.
        ),
        height=800,
        margin=dict(t=80, b=80, l=80, r=30),
        paper_bgcolor='white',
        plot_bgcolor=bg_color,
        font=dict(color='black'),
        annotations=[dict(
            text=legend_html,
            xref='paper', yref='paper',
            x=0.0, y=1.07,
            showarrow=False,
            font=dict(size=13, color='black'),
            xanchor='left',
        )],
    )
    return fig


# ── Line profiles (mean ± 95 % CI) ───────────────────────────────────────────

def make_overlay_figure(
    channel_entries: list,
    well: str,
    y_min: float | None = None,
    y_max: float | None = None,
    normalize_per_channel: bool = False,
) -> go.Figure:
    """
    Overlay mean ± 95% CI profiles for multiple channels on one plot.

    Parameters
    ----------
    channel_entries       : list of dicts:  'label', 'color', 'matrix'
    well                  : str
    y_min, y_max          : optional manual y-axis range
    normalize_per_channel : if True, normalize each channel to [0, 1] using its
                            own 1st/99th percentile before computing mean/CI.
                            Useful for comparing profile shapes across channels
                            with very different intensity scales.
    """
    traces = []

    for entry in channel_entries:
        mat = entry['matrix'].astype(np.float32)
        label = entry['label']
        color = entry['color']
        n = mat.shape[0]

        if n == 0:
            continue

        if normalize_per_channel:
            lo = float(np.nanpercentile(mat, 1))
            hi = float(np.nanpercentile(mat, 99))
            span = hi - lo
            if span > 0:
                mat = np.clip((mat - lo) / span, 0.0, 1.0)

        x = np.arange(mat.shape[1])
        mean = mat.mean(axis=0)
        ci95 = 1.96 * mat.std(axis=0, ddof=1) / np.sqrt(n)
        band_color = _hex_to_rgba(color, 0.20)

        # Upper CI — invisible anchor for fill
        traces.append(go.Scatter(
            x=x, y=mean + ci95,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
        ))
        # Lower CI filled to upper
        traces.append(go.Scatter(
            x=x, y=mean - ci95,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=band_color,
            showlegend=False,
            hoverinfo='skip',
        ))
        # Mean line
        traces.append(go.Scatter(
            x=x, y=mean,
            mode='lines',
            name=f'{label}  (n={n})',
            line=dict(color=color, width=2.5),
            hovertemplate=(
                f'<b>{label}</b><br>'
                'Position: %{x}<br>Mean: %{y:.4f}<extra></extra>'
            ),
        ))

    y_label = 'Normalized intensity (0–1)' if normalize_per_channel else 'Signal intensity'

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=f'Line profiles — {well}', font=dict(size=18, color='black')),
        xaxis=dict(
            title='Position along worm (bins)',
            title_font=dict(size=15, color='black'),
            tickfont=dict(size=13, color='black'),
            showline=True, linewidth=2, linecolor='black',
            gridcolor='#e0e0e0',
        ),
        yaxis=dict(
            title=y_label,
            title_font=dict(size=15, color='black'),
            tickfont=dict(size=13, color='black'),
            showline=True, linewidth=2, linecolor='black',
            gridcolor='#e0e0e0',
            range=[y_min, y_max] if (y_min is not None and y_max is not None) else None,
        ),
        height=500,
        margin=dict(t=60, b=80, l=90, r=30),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        legend=dict(font=dict(size=13, color='black')),
    )
    return fig
