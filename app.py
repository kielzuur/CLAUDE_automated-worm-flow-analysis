import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from parsing import (
    parse_uploaded_file, get_analyzable_columns, sort_wells,
    SPACER_OPTIONS, filter_spacer_wells, PREFERRED_ORDER as _PREF,
)
from analysis import (
    OPERATIONS, OPERATION_Y_LABEL, OUTLIER_METHODS, ROUT_Q_OPTIONS,
    get_well_mean, get_control_stats, apply_normalization,
    compute_well_stats, build_wide_dataframe,
    remove_outliers, outlier_removal_summary,
)
from plotting import make_bar_figure, make_violin_figure, make_strip_figure
from export import to_csv_bytes, make_export_filename
from profiles import (
    parse_profile_file, infer_channel_label,
    build_profile_matrix, build_aligned_matrices,
    make_profile_heatmap, make_multi_heatmap,
    make_overlay_heatmap, make_overlay_figure,
    DEFAULT_COLORS,
)

st.set_page_config(
    page_title='WormNorm',
    page_icon='🪱',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Caching ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_parse(file_bytes: bytes, file_name: str):
    df = parse_uploaded_file(file_bytes, file_name)
    cols = get_analyzable_columns(df)
    return df, cols


@st.cache_data(show_spinner='Loading profile data…')
def cached_load_profile(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    return parse_profile_file(file_bytes, file_name)


@st.cache_data(show_spinner=False)
def cached_parse_standard(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    return parse_uploaded_file(file_bytes, file_name)



def _image_dl_buttons(fig, base_name: str, key_prefix: str):
    """
    Download buttons for chart export.

    Column 1 — HTML with embedded PNG / SVG / JPEG export buttons (Plotly.js,
                browser-based, no kaleido required).
    Columns 2-4 — Direct PNG / SVG / JPEG via kaleido (if installed & working).
    """
    # ── Build the enhanced HTML (interactive chart + download toolbar) ─────────
    fig_json = fig.to_json()
    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <title>{base_name}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body{{font-family:sans-serif;margin:20px;background:#fff}}
    .dl-bar{{margin-bottom:12px}}
    .dl-bar button{{
      margin-right:8px;padding:7px 16px;
      background:#4c8bbe;color:#fff;border:none;
      border-radius:4px;cursor:pointer;font-size:14px;font-weight:600;
    }}
    .dl-bar button:hover{{background:#2e6494}}
    .dl-bar .pdf-btn{{background:#888}}
    .dl-bar .pdf-btn:hover{{background:#555}}
    @media print{{.dl-bar{{display:none}}body{{margin:0}}}}
  </style>
</head>
<body>
  <div class="dl-bar">
    <button onclick="dl('png')">&#11015; PNG</button>
    <button onclick="dl('svg')">&#11015; SVG</button>
    <button onclick="dl('jpeg')">&#11015; JPEG</button>
    <button class="pdf-btn" onclick="window.print()">&#11015; PDF</button>
  </div>
  <div id="chart"></div>
  <script>
    var fig = {fig_json};
    Plotly.newPlot('chart', fig.data, fig.layout, fig.config || {{}});
    function dl(fmt) {{
      Plotly.downloadImage('chart', {{
        format: fmt, filename: '{base_name}', width: 1400, height: 900, scale: 2
      }});
    }}
  </script>
</body>
</html>"""

    st.download_button(
        '⬇ HTML + image export',
        data=html_content.encode('utf-8'),
        file_name=f'{base_name}.html',
        mime='text/html',
        key=f'{key_prefix}_html',
        help='Opens an interactive chart with PNG / SVG / JPEG download buttons (no kaleido needed).',
    )


def _update_controls(auto_key, btn_key):
    auto = st.checkbox('Auto-update plot', value=True, key=auto_key)
    btn  = st.button('Update plot', key=btn_key, disabled=auto)
    return auto, btn


# ── Shared rendering helper ───────────────────────────────────────────────────

def render_analysis(
    src_df, param, all_wells, control_wells,
    op_key, op_label, minmax_min_well, minmax_max_well,
    file_stem, prefix,
    control_map=None,
):
    """Render plot-type selector, Raw/Normalized sub-tabs, stats expander."""
    # Recompute minmax config from src_df (important: outlier tab uses cleaned df)
    if op_key == 'minmax' and minmax_min_well and minmax_max_well:
        minmax_config = {
            'min_val': get_well_mean(src_df, minmax_min_well, param),
            'max_val': get_well_mean(src_df, minmax_max_well, param),
        }
    else:
        minmax_config = None

    df_work = src_df.copy()

    if not control_map:
        # Standard path: single pooled control for all wells
        ctrl_stats = get_control_stats(src_df, control_wells, param)
        ctrl_mean  = ctrl_stats['mean']
        if np.isnan(ctrl_mean) and op_key != 'none':
            st.warning(
                f"Control well(s) **{', '.join(control_wells)}** have no valid data "
                f"for **{param}**. Normalization cannot be applied."
            )
        norm_series, warn_msg = apply_normalization(df_work[param], ctrl_stats, op_key, minmax_config)
        if warn_msg:
            st.warning(warn_msg)
        df_work['_norm'] = norm_series
    else:
        # Per-group path: seed all rows with global control, then overwrite per-group rows
        global_ctrl_stats = get_control_stats(src_df, control_wells, param)
        ctrl_mean = global_ctrl_stats['mean']
        norm_full, warn_global = apply_normalization(df_work[param], global_ctrl_stats, op_key, minmax_config)
        if warn_global:
            st.warning(f'Global control: {warn_global}')
        df_work['_norm'] = norm_full

        seen_grp: dict = {}
        for target_well, grp_ctrl_wells in control_map.items():
            grp_key = tuple(sorted(grp_ctrl_wells))
            if grp_key not in seen_grp:
                seen_grp[grp_key] = get_control_stats(src_df, list(grp_ctrl_wells), param)
            grp_ctrl_stats = seen_grp[grp_key]
            row_mask = df_work['Source well'] == target_well
            if not row_mask.any():
                continue
            grp_series, warn_grp = apply_normalization(
                df_work.loc[row_mask, param], grp_ctrl_stats, op_key, minmax_config
            )
            if warn_grp:
                st.warning(f'Group {grp_ctrl_wells} → {target_well}: {warn_grp}')
            df_work.loc[row_mask, '_norm'] = grp_series.values

    raw_stats = compute_well_stats(src_df, param, all_wells)
    norm_stats = compute_well_stats(df_work, '_norm', all_wells)

    raw_y  = f"{param} (a.u.)"
    norm_y = OPERATION_Y_LABEL.get(op_key, op_key).replace('{param}', param)

    plot_type = st.radio(
        'Plot type',
        ['Bar + 95% CI', 'Violin', 'Scatter'],
        horizontal=True,
        key=f'{prefix}_plot_type',
    )

    if plot_type == 'Bar + 95% CI':
        raw_fig  = make_bar_figure(raw_stats,  control_wells, f'Raw {param}', raw_y)
        norm_fig = make_bar_figure(norm_stats, control_wells,
                                   f'Normalized {param} ({op_label})', norm_y)
    elif plot_type == 'Violin':
        raw_fig  = make_violin_figure(src_df,   param,   all_wells, control_wells,
                                      f'Raw {param}', raw_y)
        norm_fig = make_violin_figure(df_work, '_norm', all_wells, control_wells,
                                      f'Normalized {param} ({op_label})', norm_y)
    else:
        raw_fig  = make_strip_figure(src_df,   param,   all_wells, control_wells,
                                     f'Raw {param}', raw_y)
        norm_fig = make_strip_figure(df_work, '_norm', all_wells, control_wells,
                                     f'Normalized {param} ({op_label})', norm_y)

    sub_raw, sub_norm = st.tabs(['Raw Values', 'Normalized Values'])

    with sub_raw:
        m1, m2 = st.columns([1, 5])
        m1.metric('Wells', len(all_wells))
        ctrl_label = ', '.join(control_wells)
        m2.metric(
            'Control mean',
            f'{ctrl_mean:.4f}' if not np.isnan(ctrl_mean) else 'N/A',
            help=f'Pooled across: {ctrl_label}',
        )
        st.plotly_chart(raw_fig, use_container_width=True, height=600)
        _image_dl_buttons(raw_fig, f'{file_stem}_{param}_raw', f'{prefix}_raw_img')
        raw_wide = build_wide_dataframe(src_df, param, all_wells)
        st.download_button(
            '⬇ Download raw wide CSV',
            data=to_csv_bytes(raw_wide),
            file_name=make_export_filename(file_stem, param, 'raw', f'{prefix}_wide'),
            mime='text/csv',
            key=f'{prefix}_dl_raw',
        )

    with sub_norm:
        if op_key == 'none':
            st.info('Select a normalization operation in the sidebar to see normalized results.')
        else:
            st.plotly_chart(norm_fig, use_container_width=True, height=600)
            _image_dl_buttons(norm_fig, f'{file_stem}_{param}_{op_key}_norm', f'{prefix}_norm_img')
            norm_wide = build_wide_dataframe(df_work, '_norm', all_wells)
            norm_wide.columns = all_wells
            st.download_button(
                '⬇ Download normalized wide CSV',
                data=to_csv_bytes(norm_wide),
                file_name=make_export_filename(file_stem, param, op_key,
                                               f'{prefix}_normalized'),
                mime='text/csv',
                key=f'{prefix}_dl_norm',
            )
            if control_map:
                with st.expander('Per-group control statistics'):
                    grp_rows = []
                    unique_grp: dict = {}
                    for tw, cw_list in control_map.items():
                        k = tuple(sorted(cw_list))
                        if k not in unique_grp:
                            unique_grp[k] = get_control_stats(src_df, list(cw_list), param)
                        gs = unique_grp[k]
                        grp_rows.append({
                            'Target well': tw,
                            'Control well(s)': ', '.join(cw_list),
                            'Control mean': round(gs['mean'], 4) if not np.isnan(gs['mean']) else 'N/A',
                        })
                    st.dataframe(pd.DataFrame(grp_rows), use_container_width=True, hide_index=True)

    with st.expander('Well statistics table'):
        summary = raw_stats.rename(columns={
            'mean': f'{param}_mean',
            'ci95': f'{param}_ci95',
        })
        if op_key != 'none':
            norm_summary = norm_stats[['well', 'mean', 'ci95']].rename(columns={
                'mean': 'norm_mean',
                'ci95': 'norm_ci95',
            })
            summary = summary.merge(norm_summary, on='well')
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ── Help dialog ──────────────────────────────────────────────────────────────

@st.dialog('How to Use WormNorm', width='large')
def _show_help():
    st.markdown("""
## Overview

**WormNorm** is a browser-based tool for analyzing and normalizing flow cytometry
data from **COPAS** (Complex Object Parametric Analyzer and Sorter) worm sorters.
It reads raw per-worm measurements exported by the COPAS instrument software,
lets you define control wells, apply one or more normalization strategies, remove
outliers, and export results — without writing any code.

---

## Input Data Format

### Standard run files
WormNorm reads **tab-separated `.txt` files** produced by the COPAS instrument
software (Union Biometrica). Each file must:

- Begin with a header row whose first column is `Id`
- Contain a `Source well` column (e.g. `A1`, `B3`) identifying which plate well
  each worm came from
- Have one or more numeric measurement columns (e.g. `Green`, `Red`, `TOF`,
  `Extinction`, pulse-height/width/count variants)

Each data row represents one worm event. Instrument-generated footer text and any
rows whose first field is not a number are stripped automatically.

**Recognized measurement columns** (shown at the top of the Parameter dropdown
in preferred order):

| Column | Meaning |
|---|---|
| Green / Blue / Yellow / Red | Total fluorescence intensity per channel |
| TOF | Time of flight — proxy for body length |
| Extinction | Optical extinction — proxy for body width / optical density |
| PH Green/Blue/… | Peak height per channel |
| PW Green/Blue/… | Peak width per channel |
| PC Green/Blue/… | Peak count per channel |

Any other numeric columns in the file also appear in the parameter list.

### Profile files (Worm Profiles view only)
Profile files are a separate export from the COPAS software. They are
**wide-format** tab-separated files where:

- Each **column** is one worm event (identified by event ID)
- Each **row** is one measurement position along the worm body

Upload one file per fluorescence channel. Up to four channels are supported.

**Enabling profile saving on the COPAS instrument:**
In the COPAS software, go to **Setup → Data Storage Options** and check the
following boxes before your run:

- **Store profiles in text format** — enables profile file export
- **Store each channel in separate file** — required so each channel can be
  uploaded independently in WormNorm
- **Store oriented profiles** *(when applicable)* — saves profiles with a
  consistent head-to-tail orientation

---

## Sidebar — Setup Steps

Work through the sidebar **top-to-bottom** before switching between views. All
settings are shared across the Basic, Outlier Removed, and Worm Profiles views.

### Step 1 — Upload Files
Click **Browse files** or drag one or more `.txt` files onto the upload box. The app
parses and caches each file immediately. Re-uploading the same file byte-for-byte
is free (served from cache).

### Step 2 — File Selection
Appears only when more than one file is uploaded.

- **Select a single file** to analyze it on its own.
- **Combine all files** to pool all runs into one dataset. In combined mode, well
  names are prefixed with the first 20 characters of the file stem
  (e.g. `run1/A1`, `run2/A1`), so wells from different files remain distinct.
  Only parameters present in **all** files are available for analysis.

### Step 3 — Parameter
Select the channel or morphology parameter to analyze. Parameters are sorted by
the preferred order above; any additional numeric columns follow alphabetically.

**Parameter Math (optional):** Expand the collapsible section to create a derived
parameter on the fly by combining any two existing parameters with ÷, −, ×, or +.
Give the result a name (e.g. `Green/TOF`) and it appears at the top of the
parameter list. This is useful for size-correcting fluorescence by dividing by TOF
before normalization.

### Step 4 — Spacer Wells
Some plate layouts use structurally empty "spacer" wells that should not be
analyzed. Select the pattern that matches your plate:

| Layout option | Wells excluded |
|---|---|
| No spacers (use all wells) | None |
| Even columns are spacers | Columns 2, 4, 6, 8, … |
| Odd columns are spacers | Columns 1, 3, 5, 7, … |
| Two filled, one spacer (cols 3, 6, 9…) | Every third column |

### Step 5 — Well Labels
Optionally rename wells for display in all plots and tables. Expand the section and
type a label next to each well ID (e.g. rename `A1` → `N2 control`). Labels are
cosmetic only — the underlying data and well IDs are unchanged. Useful for
publication figures.

### Step 6 — Control Wells
Designates which well(s) serve as the normalization baseline. Two modes:

**Single / pooled control**
Pick one or more wells from the multiselect. All selected wells are pooled
together (their event-level values concatenated) to compute a single baseline
mean that is used for every well in the plate.

**Per-well mapping**
Assign different control wells to different subsets of experimental wells.
Click the *+* button to add groups; each group specifies its own control well(s)
and the target wells that will be normalized against them. Wells not assigned to
any group automatically fall back to the pooled mean of all defined control
wells. Useful when a plate has multiple internal controls (e.g. a different
strain per row).

> This section is hidden in the **Sequential Normalization** view because
> each step there has its own independent control configuration.

### Step 7 — Normalization Operation
Chooses how each per-worm value is expressed relative to the control mean. The
operation is applied to **every individual worm event** before summary statistics
(mean, CI) are computed per well.

| Operation | Per-event formula | When to use |
|---|---|---|
| None (raw values) | x | Inspect raw data, no normalization needed |
| Subtract control mean | x − μ_ctrl | Remove a constant additive background |
| Divide by control mean | x / μ_ctrl | Ratio to control; value of 1.0 = control level |
| Percent change from control | (x − μ_ctrl) / μ_ctrl × 100 | Intuitive effect size; 0% = no change |
| Log₂ ratio vs control | log₂(x / μ_ctrl) | Fold-change on a symmetric log scale; 0 = no change, +1 = doubled |
| Min-max (control range) | (x − μ_min) / (μ_max − μ_min) | Scale between two biological extremes (0 = min well, 1 = max well) |
| Multiply by control mean | x × μ_ctrl | Specialized multiplicative corrections |
| Add control mean | x + μ_ctrl | Specialized additive corrections |

**Min-max** requires two additional wells: one whose mean defines 0 (the "min"
reference) and one whose mean defines 1 (the "max" reference). These can be any
two wells on the plate — they do not have to be the same as the control wells
selected in Step 6.

---

## Analysis Views

### Basic Normalization
The standard end-to-end workflow.

1. Configure the sidebar (Steps 1–7).
2. Choose a **plot type** at the top of the main area:
   - **Bar + 95% CI** — mean bar per well with error bars (1.96 × SEM); best for
     comparing population means across many wells.
   - **Violin** — full distribution shape per well; best for seeing bimodal or
     skewed populations.
   - **Scatter** — individual worm events as dots per well; best for spotting
     outliers or sparse data.
3. The **Raw Values** sub-tab shows the unmodified channel values.
4. The **Normalized Values** sub-tab shows the result of the chosen operation.
5. The **Well statistics table** expander (bottom) shows the exact per-well mean,
   95% CI half-width, and event count used to draw the bar chart.
6. Control wells are always **highlighted in orange** across all plot types.

### Outlier Removed Normalization
Identical to Basic Normalization but with a statistical outlier-removal pass
applied to the raw data **before** normalization. Outliers are identified and
removed **per well independently** — removing a worm from one well has no effect
on any other well.

**Step 1 — Choose an outlier removal method:**

| Method | Logic | Key setting |
|---|---|---|
| Z-score | Flag events where \|z-score\| > threshold (z computed within each well) | Threshold (default **3.0**); lower = more aggressive |
| IQR | Flag events outside the range Q1 − k·IQR to Q3 + k·IQR | Multiplier k (default **1.5**); Tukey's standard; lower = more aggressive |
| Percentile cutoff | Flag events below the lower percentile or above the upper percentile | Lower % (default 2.5), upper % (default 97.5) |
| ROUT | Iterative robust outlier test (Motulsky & Brown 2006); matches GraphPad Prism's ROUT method | Max FDR Q (default **1%**); lower = more stringent |

A summary table after removal shows, per well: events before, events after, number
removed, and percent removed.

**Step 2 — Normalization** operates on the cleaned dataset exactly as in Basic
Normalization.

### Sequential Normalization
Applies **two normalization steps in series**. Step 2 operates on the numerical
output of Step 1, but its reference statistics (the control mean used in Step 2)
are always drawn from the **original raw data** — not the Step 1 output. This
means the two steps are mathematically independent in their reference values.

Both Step 1 and Step 2 have their own:
- Parameter selection (can be different channels)
- Control mode (single/pooled or per-well mapping)
- Normalization operation

**Typical use case — size-corrected percent change:**
1. Step 1: Divide Green by TOF → size-normalized fluorescence
2. Step 2: Percent change from control → express size-corrected fluorescence
   relative to the N2 wild-type wells

Results are shown in two tabs: **Step 1 Result** and **Step 2 Result (Final)**.
Both can be downloaded as CSVs independently.

### Worm Profiles
Visualizes the spatially-resolved fluorescence **profile** of individual worms —
the raw intensity measured at each position along the worm body as it passes
through the laser beam.

**Setup:**
1. Load a standard run `.txt` file in the sidebar (for well assignments and TOF).
2. In the main area, upload 1–4 **profile `.txt` files** — one per channel.
3. Assign each channel a short label and a display color.

**Shared controls (above the sub-tabs):**
- **Source well** — which well's worms to display
- **Max events** — maximum worms rendered in heatmaps (reduces rendering time for
  large wells; worms are randomly sampled if the well exceeds this cap)
- **X start / X end** — crop the body-position axis (useful to focus on the head,
  gut, or tail)

**Sub-tabs:**

| Sub-tab | Description |
|---|---|
| **Single Heatmap** | One channel at a time. Each row is one worm, sorted by TOF (body length). The X-axis is the position along the body; color encodes intensity. Adjust the color-scale min/max per channel to improve contrast. |
| **Multi-Channel Heatmap** | All channels shown as side-by-side heatmaps for the same well, with worms aligned by event ID across channels. |
| **Overlay Heatmap** | Channels blended additively in RGB (fluorescence-microscopy-style composite). Set each channel's min/max to control its contribution to the composite. Black background is recommended for additive blending. |
| **Line Profiles** | Mean ± 95% CI intensity along the worm body, with all channels overlaid as colored lines. Optional per-channel 0–1 normalization for comparing profile shapes regardless of absolute intensity. |

**Auto-update vs manual update:** The "Auto-update plot" checkbox is on by default.
Uncheck it and use the **Update plot** button if rendering is slow — the chart will
only re-render when you explicitly request it.

---

## Exporting Results

### Charts
Every plot has an **⬇ HTML + image export** button. This downloads a standalone
HTML file containing the interactive Plotly chart plus built-in download buttons for:
- **PNG** — raster image (2× scale, 1400 × 900 px default)
- **SVG** — vector image, scalable for print
- **JPEG** — smaller file size
- **PDF** — via the browser's Print → Save as PDF function

No additional software (e.g. kaleido) is required to export images from the HTML
file — all rendering happens in your browser.

### Data tables
- **⬇ Download raw wide CSV** — one column per well, one row per worm event.
  Shorter wells are NaN-padded to match the longest well.
- **⬇ Download normalized wide CSV** — same structure but with normalized values.
- **⬇ Download Step 1 / Step 2 CSV** (Sequential view) — intermediate and final
  normalized values.
- **⬇ Download matrix CSV** (Profiles → Single Heatmap) — the raw intensity matrix
  (rows = body positions, columns = event IDs) for the selected channel and well.
- **⬇ Download line profile stats CSV** (Profiles → Line Profiles) — per-position
  mean and 95% CI for each channel.

CSV filenames are auto-generated from the source file name, parameter, and
operation (e.g. `myrun_Green_pct_change_basic_normalized.csv`).

---

## Tips & Common Workflows

**Size-correcting fluorescence before normalization**
Open *Parameter Math* in Step 3, set A = `Green`, operation = ÷, B = `TOF`, name =
`Green/TOF`. Select `Green/TOF` as the parameter. In Step 7, choose Percent change
from control. This gives you size-corrected, control-normalized fluorescence in one
pass — or use Sequential Normalization for a two-step approach.

**Multiple internal controls on one plate**
Use *Per-well mapping* in Step 6. Create one group per control strain, assign the
relevant experimental wells as targets for each group. Wells with no group
assignment are normalized to the pooled mean of all defined controls.

**Comparing multiple plates / runs**
Upload all files, tick *Combine all files* in Step 2. In combined mode each well is
labeled `filestem/A1`, so wells from different plates are kept separate. You can
still select a single control well from one plate as the global baseline.

**Getting a quick QC look**
Select *None (raw values)* in Step 7. Go to **Outlier Removed Normalization** and
use the *Percentile cutoff* method with 1st–99th percentiles to trim only the most
extreme events. Compare the removal summary table to spot wells with unusually high
dropout rates.

**Comparing profile shapes across channels**
In Worm Profiles → Line Profiles, check *Normalize each channel independently
(0–1)*. This rescales each channel to its own 1st–99th percentile range, so you
can compare the spatial pattern (e.g. head-localized vs. gut-localized) regardless
of the absolute intensity differences between channels.
""")


# ── Sidebar ───────────────────────────────────────────────────────────────────

_sidebar_ready = False

with st.sidebar:
    st.title('🪱 WormNorm')
    st.markdown('---')

    active_view = st.radio(
        'Navigation',
        [
            'Basic Normalization',
            'Outlier Removed Normalization',
            'Sequential Normalization',
            'Worm Profiles',
        ],
        label_visibility='collapsed',
    )
    if st.button('📖 How to Use WormNorm', use_container_width=True):
        _show_help()
    st.markdown('---')

    # 1. Upload
    st.subheader('1. Upload Files')
    uploaded_files = st.file_uploader(
        'Upload worm sorter .txt file',
        type=['txt'],
        accept_multiple_files=True,
        help='Tab-separated COPAS/Union Biometrica format',
    )

    if not uploaded_files:
        st.info('Upload one or more files to begin.')
    else:
        parsed = {}
        for f in uploaded_files:
            file_bytes = f.read()
            try:
                with st.spinner(f'Parsing {f.name}…'):
                    df_i, cols_i = cached_parse(file_bytes, f.name)
                parsed[f.name] = (df_i, cols_i)
            except Exception as e:
                st.error(f'Could not parse **{f.name}**: {e}')
                st.stop()

        # 2. File selection (always shown)
        st.markdown('---')
        st.subheader('2. File Selection')
        if len(parsed) > 1:
            combine = st.checkbox('Combine all files', value=False)
            if not combine:
                selected_name = st.selectbox('Analyze file', list(parsed.keys()))
                active_names = [selected_name]
            else:
                active_names = list(parsed.keys())
        else:
            st.caption(f'📄 {list(parsed.keys())[0]}')
            combine = False
            active_names = list(parsed.keys())

        # Build working DataFrame
        if combine and len(active_names) > 1:
            common_cols = set.intersection(*[set(cols) for _, (_, cols) in
                                             [(n, parsed[n]) for n in active_names]])
            analyzable_cols = ([p for p in _PREF if p in common_cols] +
                               sorted(c for c in common_cols if c not in _PREF))
            dfs = []
            for name in active_names:
                df_i, _ = parsed[name]
                stem = os.path.splitext(name)[0][:20]
                df_copy = df_i.copy()
                df_copy['Source well'] = stem + '/' + df_copy['Source well'].astype(str)
                dfs.append(df_copy)
            working_df = pd.concat(dfs, ignore_index=True)
        else:
            working_df, analyzable_cols = parsed[active_names[0]]

        if not analyzable_cols:
            st.error('No analyzable numeric columns found in the selected file(s).')
            st.stop()

        # 3. Parameter
        st.markdown('---')
        st.subheader('3. Parameter')

        _PM_OPS = {
            'A ÷ B  (divide)':   'div',
            'A − B  (subtract)': 'sub',
            'A × B  (multiply)': 'mul',
            'A + B  (add)':      'add',
        }
        augmented_cols = list(analyzable_cols)
        with st.expander('Parameter Math (optional)'):
            pm_name = st.text_input('Custom name', key='pm_name', placeholder='e.g. Green/TOF')
            pm_p1 = st.selectbox('Parameter A', analyzable_cols, key='pm_p1')
            pm_op_label = st.selectbox('Operation', list(_PM_OPS.keys()), key='pm_op')
            pm_p2 = st.selectbox('Parameter B', analyzable_cols, key='pm_p2')

            pm_derived = pm_name.strip() if pm_name else ''
            if pm_derived:
                _a = working_df[pm_p1].astype(float).to_numpy()
                _b = working_df[pm_p2].astype(float).to_numpy()
                _pm_op = _PM_OPS[pm_op_label]
                if _pm_op == 'div':
                    with np.errstate(divide='ignore', invalid='ignore'):
                        _pm_result = np.where(_b != 0, _a / _b, np.nan)
                elif _pm_op == 'sub':
                    _pm_result = _a - _b
                elif _pm_op == 'mul':
                    _pm_result = _a * _b
                else:
                    _pm_result = _a + _b
                working_df = working_df.copy()
                working_df[pm_derived] = _pm_result
                augmented_cols = [pm_derived] + augmented_cols
                st.caption(f'✓ "{pm_derived}" added as selectable parameter.')

        param = st.selectbox('Channel / parameter', augmented_cols, index=0)

        # 4. Spacer wells
        all_wells_raw = sort_wells(working_df['Source well'].dropna().unique())
        st.markdown('---')
        st.subheader('4. Spacer Wells')
        spacer_label = st.selectbox(
            'Plate layout',
            list(SPACER_OPTIONS.keys()),
            index=0,
            help='Exclude structural spacer wells from the plate layout.',
        )
        spacer_pattern = SPACER_OPTIONS[spacer_label]
        all_wells = filter_spacer_wells(all_wells_raw, spacer_pattern)
        if not all_wells:
            st.error('All wells excluded by spacer filter. Choose a different layout.')
            st.stop()
        if len(all_wells) < len(all_wells_raw):
            st.caption(f'{len(all_wells_raw) - len(all_wells)} spacer well(s) excluded.')

        # 5. Well Labels
        st.markdown('---')
        st.subheader('5. Well Labels')
        _well_labels = {w: w for w in all_wells}
        with st.expander('Rename wells for display (optional)'):
            st.caption('Labels replace well IDs in all plots and tables.')
            for _w in all_wells:
                _lbl = st.text_input(_w, value=_w, key=f'wlabel_{_w}')
                _well_labels[_w] = _lbl.strip() if _lbl.strip() else _w

        # Apply labels to the DataFrame and well lists
        if any(v != k for k, v in _well_labels.items()):
            working_df = working_df.copy()
            working_df['Source well'] = working_df['Source well'].map(
                lambda x: _well_labels.get(x, x)
            )
        all_wells = [_well_labels[w] for w in all_wells]

        if active_view != 'Sequential Normalization':
            # 6. Control wells
            st.markdown('---')
            st.subheader('6. Control Wells')

            ctrl_mode = st.radio(
                'Control mode',
                ['Single / pooled control', 'Per-well mapping'],
                horizontal=True,
                key='ctrl_mode',
                label_visibility='collapsed',
            )

            control_wells = []
            control_map = {}

            # Preserve per-well mapping config across mode switches.
            # Streamlit clears widget keys when their widgets aren't rendered, so we
            # save to non-widget keys on the way out and restore on the way back in.
            _prev_ctrl_mode = st.session_state.get('_prev_ctrl_mode')
            if _prev_ctrl_mode == 'Per-well mapping' and ctrl_mode == 'Single / pooled control':
                # Switching away: snapshot group config before widget keys get wiped
                _snap_n = int(st.session_state.get('n_ctrl_groups', 1))
                st.session_state['_saved_n_ctrl_groups'] = _snap_n
                for _g in range(_snap_n):
                    st.session_state[f'_saved_grp_ctrl_{_g}'] = st.session_state.get(f'grp_ctrl_{_g}', [])
                    st.session_state[f'_saved_grp_targets_{_g}'] = st.session_state.get(f'grp_targets_{_g}', [])
            elif _prev_ctrl_mode == 'Single / pooled control' and ctrl_mode == 'Per-well mapping':
                # Switching back: inject saved values into widget keys before they render
                _snap_n = st.session_state.get('_saved_n_ctrl_groups', 1)
                st.session_state['n_ctrl_groups'] = _snap_n
                for _g in range(_snap_n):
                    _sc = st.session_state.get(f'_saved_grp_ctrl_{_g}')
                    _st = st.session_state.get(f'_saved_grp_targets_{_g}')
                    if _sc is not None:
                        st.session_state[f'grp_ctrl_{_g}'] = _sc
                    if _st is not None:
                        st.session_state[f'grp_targets_{_g}'] = _st
            st.session_state['_prev_ctrl_mode'] = ctrl_mode

            if ctrl_mode == 'Single / pooled control':
                control_wells = st.multiselect(
                    'Control well(s)',
                    all_wells,
                    default=[all_wells[0]] if all_wells else [],
                    help='Select one or more wells to pool as the normalization baseline.',
                )
                if not control_wells:
                    st.warning('Select at least one control well.')
                    st.stop()

            else:
                # Per-well mapping — no global control required
                st.caption(
                    'Each group\'s control wells normalize its target wells. '
                    'Wells not in any group use the pooled control across all groups as fallback.'
                )
                n_groups = st.number_input(
                    'Number of groups', min_value=1, max_value=20,
                    value=1, step=1, key='n_ctrl_groups',
                )

                # Pre-collect all control wells selected across groups to exclude from target options
                _all_grp_ctrls: set = set()
                for _g in range(int(n_groups)):
                    for _cw in st.session_state.get(f'grp_ctrl_{_g}', []):
                        _all_grp_ctrls.add(_cw)
                available_targets = [w for w in all_wells if w not in _all_grp_ctrls]

                all_mapped_targets = []
                _grp_defs = []
                for g in range(int(n_groups)):
                    st.markdown(f'**Group {g + 1}**')
                    gcol1, gcol2 = st.columns(2)
                    with gcol1:
                        grp_ctrl = st.multiselect('Control well(s)', all_wells, key=f'grp_ctrl_{g}')
                    with gcol2:
                        grp_targets = st.multiselect('Target well(s)', available_targets, key=f'grp_targets_{g}')
                    if grp_ctrl and grp_targets:
                        for tw in grp_targets:
                            control_map[tw] = grp_ctrl
                        all_mapped_targets.extend(grp_targets)
                        _grp_defs.append(grp_ctrl)

                # Auto-include each group's control wells as targets of their own group
                for _gc in _grp_defs:
                    for cw in _gc:
                        if cw not in control_map:
                            control_map[cw] = _gc

                # Fallback control = union of all defined control wells
                control_wells = list({cw for _gc in _grp_defs for cw in _gc})

                # Warn about duplicate target assignments
                seen_counts: dict = {}
                for tw in all_mapped_targets:
                    seen_counts[tw] = seen_counts.get(tw, 0) + 1
                dupes = [w for w, c in seen_counts.items() if c > 1]
                if dupes:
                    st.warning(
                        f'These wells appear in multiple groups (last assignment wins): '
                        f'{", ".join(dupes)}'
                    )
                unmapped = [w for w in all_wells if w not in control_map]
                if unmapped and control_wells:
                    st.caption(
                        f'{len(unmapped)} well(s) not in any group use pooled fallback: '
                        f'{", ".join(unmapped)}'
                    )
                elif not control_wells:
                    st.caption('Define at least one group with control and target wells to enable normalization.')

            # 7. Normalization
            st.markdown('---')
            st.subheader('7. Normalization')
            op_label = st.selectbox(
                'Operation vs control',
                list(OPERATIONS.keys()),
                index=0,
            )
            op_key = OPERATIONS[op_label]

            minmax_min_well = None
            minmax_max_well = None
            if op_key == 'minmax':
                st.markdown('**Min-max reference wells**')
                mm1, mm2 = st.columns(2)
                with mm1:
                    minmax_min_well = st.selectbox(
                        'Min well', all_wells, index=0, key='mm_min_well',
                        help='Well whose mean is used as the minimum reference.',
                    )
                with mm2:
                    minmax_max_well = st.selectbox(
                        'Max well', all_wells, index=len(all_wells) - 1, key='mm_max_well',
                        help='Well whose mean is used as the maximum reference.',
                    )

            st.markdown('---')
            st.caption('These settings apply to the Basic and Outlier tabs.')
        else:
            # Provide safe defaults so the rest of the script doesn't crash
            control_wells = []
            control_map = {}
            op_label = 'None (raw values)'
            op_key = 'none'
            minmax_min_well = None
            minmax_max_well = None

        _sidebar_ready = True

# ── File stem for exports ─────────────────────────────────────────────────────

file_stem = (
    ('combined' if (combine and len(active_names) > 1) else os.path.splitext(active_names[0])[0])
    if _sidebar_ready else 'export'
)

# ── Main area ─────────────────────────────────────────────────────────────────

st.title('WormNorm')
st.subheader(active_view)
st.markdown('---')

# ── Basic Normalization ───────────────────────────────────────────────────────

if active_view == 'Basic Normalization':
    if not _sidebar_ready:
        st.info('Upload one or more .txt files in the sidebar to use this tab.')
    else:
        render_analysis(
            src_df=working_df,
            param=param,
            all_wells=all_wells,
            control_wells=control_wells,
            op_key=op_key,
            op_label=op_label,
            minmax_min_well=minmax_min_well,
            minmax_max_well=minmax_max_well,
            file_stem=file_stem,
            prefix='basic',
            control_map=control_map or None,
        )

# ── Outlier Removed Normalization ────────────────────────────────────────────

elif active_view == 'Outlier Removed Normalization':
    if not _sidebar_ready:
        st.info('Upload one or more .txt files in the sidebar to use this tab.')
    else:
        st.subheader('Step 1 — Outlier Removal')

        oc1, oc2, oc3 = st.columns([2, 2, 2])

        with oc1:
            outlier_method_label = st.selectbox(
                'Method',
                list(OUTLIER_METHODS.keys()),
                key='ol_method',
            )
        outlier_method = OUTLIER_METHODS[outlier_method_label]

        outlier_kwargs = {}
        with oc2:
            if outlier_method == 'zscore':
                thr = st.slider('Z-score threshold', 1.5, 5.0, 3.0, 0.1, key='ol_zscore')
                outlier_kwargs['threshold'] = thr
                st.caption(f'Remove |z| > {thr}')
            elif outlier_method == 'iqr':
                k = st.slider('IQR multiplier (k)', 0.5, 5.0, 1.5, 0.25, key='ol_iqr')
                outlier_kwargs['k'] = k
                st.caption(f'Remove outside Q1 − {k}·IQR  to  Q3 + {k}·IQR')
            elif outlier_method == 'percentile':
                lo_pct = st.number_input('Lower percentile', 0.0, 20.0, 2.5, 0.5, key='ol_lo')
                outlier_kwargs['lo_pct'] = lo_pct
            elif outlier_method == 'rout':
                q_label = st.selectbox('Maximum FDR (Q)', list(ROUT_Q_OPTIONS.keys()),
                                       index=1, key='ol_rout_q')
                outlier_kwargs['Q'] = ROUT_Q_OPTIONS[q_label]
                st.caption('Iterative Grubbs test on robust residuals (median + scaled MAD). '
                           'Q = max false discovery rate.')
        with oc3:
            if outlier_method == 'percentile':
                hi_pct = st.number_input('Upper percentile', 80.0, 100.0, 97.5, 0.5, key='ol_hi')
                outlier_kwargs['hi_pct'] = hi_pct

        cleaned_df = remove_outliers(working_df, param, all_wells, outlier_method, **outlier_kwargs)

        summary_df = outlier_removal_summary(working_df, cleaned_df, all_wells)
        total_removed = summary_df['n_removed'].sum()
        total_before  = summary_df['n_before'].sum()
        pct_total = round(total_removed / total_before * 100, 2) if total_before > 0 else 0.0

        sm1, sm2, sm3 = st.columns(3)
        sm1.metric('Events before', f'{total_before:,}')
        sm2.metric('Events after',  f'{total_before - total_removed:,}')
        sm3.metric('Removed',       f'{total_removed:,}  ({pct_total}%)')

        with st.expander('Outlier removal summary per well'):
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown('---')
        st.subheader('Step 2 — Normalization')

        render_analysis(
            src_df=cleaned_df,
            param=param,
            all_wells=all_wells,
            control_wells=control_wells,
            op_key=op_key,
            op_label=op_label,
            minmax_min_well=minmax_min_well,
            minmax_max_well=minmax_max_well,
            file_stem=file_stem,
            prefix='outlier',
            control_map=control_map or None,
        )

# ── Sequential Normalization ──────────────────────────────────────────────────

elif active_view == 'Sequential Normalization':
    if not _sidebar_ready:
        st.info('Upload one or more .txt files in the sidebar to use this tab.')
    else:
        st.caption(
            'Apply two normalization steps in sequence. '
            'Step 2 operates on the output of Step 1. '
            'Each step can use a different parameter and control well selection. '
            'Reference statistics for Step 2 are always drawn from the raw data.'
        )

        sc1, sc2 = st.columns(2)

        with sc1:
            st.markdown('### Step 1')
            seq_param1 = st.selectbox('Parameter', augmented_cols, key='seq_param1')

            seq1_ctrl_mode = st.radio(
                'Step 1 control mode',
                ['Single / pooled control', 'Per-well mapping'],
                horizontal=True, key='seq1_ctrl_mode', label_visibility='collapsed',
            )
            seq_op_label1 = st.selectbox('Operation', list(OPERATIONS.keys()), key='seq_op1')
            seq_op_key1 = OPERATIONS[seq_op_label1]

            seq_ctrl1 = []
            seq_map1 = {}

            _prev_s1 = st.session_state.get('_prev_seq1_ctrl_mode')
            if _prev_s1 == 'Per-well mapping' and seq1_ctrl_mode == 'Single / pooled control':
                _snap = int(st.session_state.get('seq1_n_grp_main', 1))
                st.session_state['_saved_seq1_n'] = _snap
                for _g in range(_snap):
                    st.session_state[f'_saved_seq1_ctrl_{_g}'] = st.session_state.get(f'seq1_main_ctrl_{_g}', [])
                    st.session_state[f'_saved_seq1_tgt_{_g}'] = st.session_state.get(f'seq1_main_tgt_{_g}', [])
            elif _prev_s1 == 'Single / pooled control' and seq1_ctrl_mode == 'Per-well mapping':
                _snap = st.session_state.get('_saved_seq1_n', 1)
                st.session_state['seq1_n_grp_main'] = _snap
                for _g in range(_snap):
                    _sc = st.session_state.get(f'_saved_seq1_ctrl_{_g}')
                    _st = st.session_state.get(f'_saved_seq1_tgt_{_g}')
                    if _sc is not None:
                        st.session_state[f'seq1_main_ctrl_{_g}'] = _sc
                    if _st is not None:
                        st.session_state[f'seq1_main_tgt_{_g}'] = _st
            st.session_state['_prev_seq1_ctrl_mode'] = seq1_ctrl_mode

            if seq1_ctrl_mode == 'Single / pooled control':
                seq_ctrl1 = st.multiselect(
                    'Control well(s)', all_wells,
                    default=[all_wells[0]] if all_wells else [],
                    key='seq_ctrl1',
                )
            else:
                st.caption("Each group's control wells normalize its target wells.")
                n_s1 = st.number_input('Number of groups', min_value=1, max_value=20,
                                       value=1, step=1, key='seq1_n_grp_main')
                _s1_all_ctrls: set = set()
                for _g in range(int(n_s1)):
                    for _cw in st.session_state.get(f'seq1_main_ctrl_{_g}', []):
                        _s1_all_ctrls.add(_cw)
                s1_avail_tgts = [w for w in all_wells if w not in _s1_all_ctrls]
                _s1_grp_defs = []
                for g in range(int(n_s1)):
                    st.markdown(f'**Group {g + 1}**')
                    s1gc = st.multiselect('Control well(s)', all_wells, key=f'seq1_main_ctrl_{g}')
                    s1gt = st.multiselect('Target well(s)', s1_avail_tgts, key=f'seq1_main_tgt_{g}')
                    if s1gc and s1gt:
                        for tw in s1gt:
                            seq_map1[tw] = s1gc
                        _s1_grp_defs.append(s1gc)
                for _gc in _s1_grp_defs:
                    for cw in _gc:
                        if cw not in seq_map1:
                            seq_map1[cw] = _gc
                seq_ctrl1 = list({cw for _gc in _s1_grp_defs for cw in _gc})
                if not seq_ctrl1:
                    st.caption('Define at least one group to enable Step 1 normalization.')

        with sc2:
            st.markdown('### Step 2')
            seq_param2 = st.selectbox(
                'Reference parameter', augmented_cols, key='seq_param2',
                help='Parameter used to compute the Step 2 reference value from the raw data.',
            )

            seq2_ctrl_mode = st.radio(
                'Step 2 control mode',
                ['Single / pooled control', 'Per-well mapping'],
                horizontal=True, key='seq2_ctrl_mode', label_visibility='collapsed',
            )
            seq_op_label2 = st.selectbox('Operation', list(OPERATIONS.keys()), key='seq_op2')
            seq_op_key2 = OPERATIONS[seq_op_label2]

            seq_ctrl2 = []
            seq_map2 = {}

            _prev_s2 = st.session_state.get('_prev_seq2_ctrl_mode')
            if _prev_s2 == 'Per-well mapping' and seq2_ctrl_mode == 'Single / pooled control':
                _snap = int(st.session_state.get('seq2_n_grp_main', 1))
                st.session_state['_saved_seq2_n'] = _snap
                for _g in range(_snap):
                    st.session_state[f'_saved_seq2_ctrl_{_g}'] = st.session_state.get(f'seq2_main_ctrl_{_g}', [])
                    st.session_state[f'_saved_seq2_tgt_{_g}'] = st.session_state.get(f'seq2_main_tgt_{_g}', [])
            elif _prev_s2 == 'Single / pooled control' and seq2_ctrl_mode == 'Per-well mapping':
                _snap = st.session_state.get('_saved_seq2_n', 1)
                st.session_state['seq2_n_grp_main'] = _snap
                for _g in range(_snap):
                    _sc = st.session_state.get(f'_saved_seq2_ctrl_{_g}')
                    _st = st.session_state.get(f'_saved_seq2_tgt_{_g}')
                    if _sc is not None:
                        st.session_state[f'seq2_main_ctrl_{_g}'] = _sc
                    if _st is not None:
                        st.session_state[f'seq2_main_tgt_{_g}'] = _st
            st.session_state['_prev_seq2_ctrl_mode'] = seq2_ctrl_mode

            if seq2_ctrl_mode == 'Single / pooled control':
                seq_ctrl2 = st.multiselect(
                    'Reference well(s)', all_wells,
                    default=[all_wells[0]] if all_wells else [],
                    key='seq_ctrl2',
                )
            else:
                st.caption("Each group's reference wells are used as the Step 2 baseline.")
                n_s2 = st.number_input('Number of groups', min_value=1, max_value=20,
                                       value=1, step=1, key='seq2_n_grp_main')
                _s2_all_ctrls: set = set()
                for _g in range(int(n_s2)):
                    for _cw in st.session_state.get(f'seq2_main_ctrl_{_g}', []):
                        _s2_all_ctrls.add(_cw)
                s2_avail_tgts = [w for w in all_wells if w not in _s2_all_ctrls]
                _s2_grp_defs = []
                for g in range(int(n_s2)):
                    st.markdown(f'**Group {g + 1}**')
                    s2gc = st.multiselect('Reference well(s)', all_wells, key=f'seq2_main_ctrl_{g}')
                    s2gt = st.multiselect('Target well(s)', s2_avail_tgts, key=f'seq2_main_tgt_{g}')
                    if s2gc and s2gt:
                        for tw in s2gt:
                            seq_map2[tw] = s2gc
                        _s2_grp_defs.append(s2gc)
                for _gc in _s2_grp_defs:
                    for cw in _gc:
                        if cw not in seq_map2:
                            seq_map2[cw] = _gc
                seq_ctrl2 = list({cw for _gc in _s2_grp_defs for cw in _gc})
                if not seq_ctrl2:
                    st.caption('Define at least one group to enable Step 2 normalization.')

        if not seq_ctrl1 or not seq_ctrl2:
            st.warning('Select control well(s) for both steps.')
        else:
            df_seq = working_df.copy()

            # ── Step 1 ────────────────────────────────────────────────────────
            seq_ctrl_stats1 = get_control_stats(working_df, seq_ctrl1, seq_param1)
            seq_norm1_full, seq_warn1_g = apply_normalization(
                working_df[seq_param1], seq_ctrl_stats1, seq_op_key1, None
            )
            if seq_warn1_g:
                st.warning(f'Step 1 global: {seq_warn1_g}')
            df_seq['_seq1'] = seq_norm1_full

            if seq_map1:
                _seen_cs1: dict = {}
                for tw, cw_list in seq_map1.items():
                    k = tuple(sorted(cw_list))
                    if k not in _seen_cs1:
                        _seen_cs1[k] = get_control_stats(working_df, list(cw_list), seq_param1)
                    mask = df_seq['Source well'] == tw
                    if not mask.any():
                        continue
                    grp_n, grp_w = apply_normalization(
                        df_seq.loc[mask, seq_param1], _seen_cs1[k], seq_op_key1, None
                    )
                    if grp_w:
                        st.warning(f'Step 1 → {tw}: {grp_w}')
                    df_seq.loc[mask, '_seq1'] = grp_n.values

            # ── Step 2 (reference stats always from raw data) ─────────────────
            seq_ctrl_stats2 = get_control_stats(working_df, seq_ctrl2, seq_param2)
            seq_norm2_full, seq_warn2_g = apply_normalization(
                df_seq['_seq1'], seq_ctrl_stats2, seq_op_key2, None
            )
            if seq_warn2_g:
                st.warning(f'Step 2 global: {seq_warn2_g}')
            df_seq['_seq2'] = seq_norm2_full

            if seq_map2:
                _seen_cs2: dict = {}
                for tw, cw_list in seq_map2.items():
                    k = tuple(sorted(cw_list))
                    if k not in _seen_cs2:
                        _seen_cs2[k] = get_control_stats(working_df, list(cw_list), seq_param2)
                    mask = df_seq['Source well'] == tw
                    if not mask.any():
                        continue
                    grp_n, grp_w = apply_normalization(
                        df_seq.loc[mask, '_seq1'], _seen_cs2[k], seq_op_key2, None
                    )
                    if grp_w:
                        st.warning(f'Step 2 → {tw}: {grp_w}')
                    df_seq.loc[mask, '_seq2'] = grp_n.values

            # ── Display ───────────────────────────────────────────────────────
            seq_stats1 = compute_well_stats(df_seq, '_seq1', all_wells)
            seq_stats2 = compute_well_stats(df_seq, '_seq2', all_wells)

            step1_y = OPERATION_Y_LABEL.get(seq_op_key1, seq_op_key1).replace('{param}', seq_param1)
            step2_y = OPERATION_Y_LABEL.get(seq_op_key2, seq_op_key2).replace('{param}', f'Step1({seq_param1})')

            seq_plot_type = st.radio(
                'Plot type',
                ['Bar + 95% CI', 'Violin', 'Scatter'],
                horizontal=True,
                key='seq_plot_type',
            )

            tab_s1, tab_s2 = st.tabs(['Step 1 Result', 'Step 2 Result (Final)'])

            with tab_s1:
                m1, m2 = st.columns([1, 5])
                m1.metric('Wells', len(all_wells))
                cm1 = seq_ctrl_stats1['mean']
                m2.metric(
                    'Step 1 control mean',
                    f'{cm1:.4f}' if not np.isnan(cm1) else 'N/A',
                    help=f'Pooled across: {", ".join(seq_ctrl1)}',
                )
                step1_title = f'Step 1: {seq_op_label1}({seq_param1})'
                if seq_plot_type == 'Bar + 95% CI':
                    fig_s1 = make_bar_figure(seq_stats1, seq_ctrl1, step1_title, step1_y)
                elif seq_plot_type == 'Violin':
                    fig_s1 = make_violin_figure(df_seq, '_seq1', all_wells, seq_ctrl1, step1_title, step1_y)
                else:
                    fig_s1 = make_strip_figure(df_seq, '_seq1', all_wells, seq_ctrl1, step1_title, step1_y)
                st.plotly_chart(fig_s1, use_container_width=True, height=600)
                _image_dl_buttons(fig_s1, f'{file_stem}_{seq_param1}_{seq_op_key1}_seq1', 'seq_img1')
                s1_wide = build_wide_dataframe(df_seq, '_seq1', all_wells)
                s1_wide.columns = all_wells
                st.download_button(
                    '⬇ Download Step 1 CSV',
                    data=to_csv_bytes(s1_wide),
                    file_name=make_export_filename(file_stem, seq_param1, seq_op_key1, 'seq_step1'),
                    mime='text/csv',
                    key='seq_dl1',
                )
                if seq_map1 and seq_op_key1 != 'none':
                    with st.expander('Per-group Step 1 statistics'):
                        rows = []
                        uniq: dict = {}
                        for tw, cw_list in seq_map1.items():
                            k = tuple(sorted(cw_list))
                            if k not in uniq:
                                uniq[k] = get_control_stats(working_df, list(cw_list), seq_param1)
                            gs = uniq[k]
                            rows.append({'Target well': tw, 'Control well(s)': ', '.join(cw_list),
                                         'Control mean': round(gs['mean'], 4) if not np.isnan(gs['mean']) else 'N/A'})
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            with tab_s2:
                m1, m2 = st.columns([1, 5])
                m1.metric('Wells', len(all_wells))
                cm2 = seq_ctrl_stats2['mean']
                m2.metric(
                    'Step 2 reference mean',
                    f'{cm2:.4f}' if not np.isnan(cm2) else 'N/A',
                    help=f'{seq_param2} pooled across: {", ".join(seq_ctrl2)}',
                )
                step2_title = f'Step 2: {seq_op_label2}(Step1, ref={seq_param2})'
                if seq_plot_type == 'Bar + 95% CI':
                    fig_s2 = make_bar_figure(seq_stats2, seq_ctrl1, step2_title, step2_y)
                elif seq_plot_type == 'Violin':
                    fig_s2 = make_violin_figure(df_seq, '_seq2', all_wells, seq_ctrl1, step2_title, step2_y)
                else:
                    fig_s2 = make_strip_figure(df_seq, '_seq2', all_wells, seq_ctrl1, step2_title, step2_y)
                st.plotly_chart(fig_s2, use_container_width=True, height=600)
                _image_dl_buttons(fig_s2, f'{file_stem}_{seq_param1}_{seq_op_key1}_{seq_op_key2}_seq2', 'seq_img2')
                s2_wide = build_wide_dataframe(df_seq, '_seq2', all_wells)
                s2_wide.columns = all_wells
                st.download_button(
                    '⬇ Download Step 2 (Final) CSV',
                    data=to_csv_bytes(s2_wide),
                    file_name=make_export_filename(
                        file_stem, seq_param1, f'{seq_op_key1}_{seq_op_key2}', 'seq_final'
                    ),
                    mime='text/csv',
                    key='seq_dl2',
                )
                if seq_map2 and seq_op_key2 != 'none':
                    with st.expander('Per-group Step 2 statistics'):
                        rows = []
                        uniq: dict = {}
                        for tw, cw_list in seq_map2.items():
                            k = tuple(sorted(cw_list))
                            if k not in uniq:
                                uniq[k] = get_control_stats(working_df, list(cw_list), seq_param2)
                            gs = uniq[k]
                            rows.append({'Target well': tw, 'Reference well(s)': ', '.join(cw_list),
                                         'Reference mean': round(gs['mean'], 4) if not np.isnan(gs['mean']) else 'N/A'})
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            with st.expander('Well statistics table'):
                s1_summary = seq_stats1.rename(columns={'mean': 'step1_mean', 'ci95': 'step1_ci95'})
                s2_summary = seq_stats2[['well', 'mean', 'ci95']].rename(
                    columns={'mean': 'step2_mean', 'ci95': 'step2_ci95'}
                )
                st.dataframe(
                    s1_summary.merge(s2_summary, on='well'),
                    use_container_width=True,
                    hide_index=True,
                )

# ── Worm Profiles ────────────────────────────────────────────────────────────

def _render_profiles_tab():
    """Profiles tab body; uses return instead of st.stop() so other tabs are unaffected."""
    st.subheader('Upload Profile Data')
    st.markdown('**Profile files** — upload up to 4 channels')
    prf_cols_ui = st.columns(4)
    prf_uploads = []
    for i, col in enumerate(prf_cols_ui):
        with col:
            f = st.file_uploader(f'Channel {i + 1}', type=['txt'], key=f'prf_{i}')
            default_label = infer_channel_label(f.name) if f else f'Channel {i + 1}'
            label = st.text_input('Label', value=default_label, key=f'prf_label_{i}')
            color = st.color_picker('Color', value=DEFAULT_COLORS[i], key=f'prf_color_{i}')
            prf_uploads.append({'file': f, 'label': label, 'color': color})

    active_channels = [p for p in prf_uploads if p['file'] is not None]

    if not active_channels:
        st.info('Upload at least one profile file above to begin.')
        return

    # Parse all active profile files
    profile_dfs = {}
    for ch in active_channels:
        try:
            fb = ch['file'].read()
            ch['file'].seek(0)
            profile_dfs[ch['label']] = cached_load_profile(fb, ch['file'].name)
        except Exception as e:
            st.error(f"Could not parse profile file '{ch['file'].name}': {e}")
            return

    # Compute global 1st/99th percentile per channel (sampled from entire file)
    _global_ranges = {}
    _rng_seed = np.random.default_rng(seed=0)
    for _ch in active_channels:
        _gr_key = f'global_range|{_ch["file"].name}:{_ch["file"].size}'
        if _gr_key not in st.session_state:
            _df = profile_dfs[_ch['label']]
            _n = len(_df.columns)
            _idx = _rng_seed.choice(_n, min(500, _n), replace=False)
            _sample = _df.iloc[:, _idx].to_numpy(dtype=np.float32).ravel()
            st.session_state[_gr_key] = (
                float(np.nanpercentile(_sample, 1)),
                float(np.nanpercentile(_sample, 99)),
            )
        _global_ranges[_ch['label']] = st.session_state[_gr_key]

    if not _sidebar_ready:
        st.info('Upload a standard data file in the sidebar to enable well selection.')
        return

    # Build source-well lookup from the already-loaded sidebar data
    id_to_well = working_df.set_index('Id')['Source well'].to_dict()
    id_to_tof  = working_df.set_index('Id')['TOF'].to_dict() if 'TOF' in working_df.columns else {}
    prf_wells  = sort_wells(working_df['Source well'].dropna().unique())

    # ── Shared controls ──
    st.markdown('---')
    sc1, sc2, sc3, sc4 = st.columns([2, 2, 1, 1])
    with sc1:
        prf_well = st.selectbox('Source well', prf_wells, key='prf_well')
    with sc2:
        max_events = st.slider('Max events (heatmap)', 50, 2000, 500, 50, key='prf_max_ev')
    with sc3:
        x_start = st.number_input('X start', value=None, placeholder='auto',
                                  min_value=0, key='prf_xstart')
    with sc4:
        x_end = st.number_input('X end', value=None, placeholder='auto',
                                min_value=0, key='prf_xend')
    prf_x_range = [x_start, x_end] if (x_start is not None or x_end is not None) else None

    well_ids = [int(eid) for eid, w in id_to_well.items() if w == prf_well]
    tof_series = pd.Series(id_to_tof)

    if not well_ids:
        st.warning(f'No events found for well {prf_well} in the loaded data file.')
        return

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    sub_single, sub_multi, sub_ovhm, sub_line = st.tabs([
        'Single Heatmap', 'Multi-Channel Heatmap', 'Overlay Heatmap', 'Line Profiles',
    ])

    def _get_single_matrix(ch_label, n_events_cap):
        ch_obj = next(c for c in active_channels if c['label'] == ch_label)
        key = f'hm_mat|{ch_obj["file"].name}:{ch_obj["file"].size}|{prf_well}|{n_events_cap}'
        if key not in st.session_state:
            st.session_state[key] = build_profile_matrix(
                profile_dfs[ch_label], well_ids, tof_series, n_events_cap
            )
        return st.session_state[key]

    def _get_aligned_matrices(n_events_cap):
        meta = '|'.join(f'{c["file"].name}:{c["file"].size}' for c in active_channels)
        key = f'aligned_mat|{meta}|{prf_well}|{n_events_cap}'
        if key not in st.session_state:
            prf_dfs_subset = {c['label']: profile_dfs[c['label']] for c in active_channels}
            st.session_state[key] = build_aligned_matrices(
                prf_dfs_subset, well_ids, tof_series, n_events_cap
            )
        return st.session_state[key]

    def _vrange_inputs(label, default_lo, default_hi, prefix):
        vmin_k, vmax_k = f'{prefix}_vmin_{label}', f'{prefix}_vmax_{label}'
        if vmin_k not in st.session_state:
            st.session_state[vmin_k] = float(round(default_lo))
        if vmax_k not in st.session_state:
            st.session_state[vmax_k] = float(round(default_hi))
        c1, c2 = st.columns(2)
        with c1:
            vmin = st.number_input(f'{label} — color min', key=vmin_k, step=10.0)
        with c2:
            vmax = st.number_input(f'{label} — color max', key=vmax_k, step=10.0)
        return vmin, vmax

    # ── 1. Single Heatmap ─────────────────────────────────────────────────────
    with sub_single:
        channel_labels = [ch['label'] for ch in active_channels]
        hm_channel = st.selectbox('Channel to display', channel_labels, key='prf_hm_ch')

        matrix, sorted_ids, sorted_tofs = _get_single_matrix(hm_channel, max_events)

        if matrix.size == 0:
            st.warning(f'No matching events found in "{hm_channel}" profile for well {prf_well}.')
        else:
            hm_c1, hm_c2, hm_c3 = st.columns([2, 2, 2])
            with hm_c1:
                vmin, vmax = _vrange_inputs(hm_channel, *_global_ranges[hm_channel], 'sng')
            with hm_c2:
                bg_color = st.radio(
                    'Background color', ['White', 'Black'],
                    horizontal=True, key='prf_bg',
                ).lower()
            with hm_c3:
                sng_auto, sng_btn = _update_controls('sng_auto', 'sng_btn')

            hm_channel_color = next(
                ch['color'] for ch in active_channels if ch['label'] == hm_channel
            )
            if sng_auto or sng_btn:
                _fig = make_profile_heatmap(
                    matrix, sorted_tofs, prf_well, hm_channel,
                    vmin, vmax, hm_channel_color, bg_color,
                )
                if prf_x_range:
                    _fig.update_xaxes(range=prf_x_range)
                st.session_state['prf_sng_fig'] = _fig

            cached = st.session_state.get('prf_sng_fig')
            if cached is not None:
                st.plotly_chart(cached, use_container_width=True, height=800)
                _image_dl_buttons(cached, f'profile_{hm_channel}_{prf_well}', 'sng_img')
            else:
                st.info('Click "Update plot" to render.')

            m1, m2, m3 = st.columns(3)
            m1.metric('Events displayed', len(sorted_ids))
            m2.metric('Total events in well', len(well_ids))
            tof_vals_valid = [t for t in sorted_tofs if t > 0]
            m3.metric('TOF range',
                      f'{min(tof_vals_valid)} – {max(tof_vals_valid)}' if tof_vals_valid else 'N/A')

            hm_export = pd.DataFrame(
                matrix,
                index=[str(eid) for eid in sorted_ids],
                columns=[str(p) for p in range(matrix.shape[1])],
            )
            hm_export.index.name = 'event_id'
            st.download_button(
                '⬇ Download matrix CSV',
                data=to_csv_bytes(hm_export.reset_index()),
                file_name=f'profile_{hm_channel}_{prf_well}.csv',
                mime='text/csv',
                key='prf_dl_sng',
            )

    # ── 2. Multi-Channel Heatmap ──────────────────────────────────────────────
    with sub_multi:
        if len(active_channels) < 2:
            st.info('Upload at least two channel files to use this view.')
        else:
            matrices_aligned, aligned_ids, aligned_tofs = _get_aligned_matrices(max_events)

            if not aligned_ids:
                st.warning(f'No events common to all channels for well {prf_well}.')
            else:
                st.markdown('**Per-channel color range**')
                multi_entries = []
                for ch in active_channels:
                    mat = matrices_aligned.get(ch['label'])
                    if mat is None or mat.size == 0:
                        continue
                    vmin_m, vmax_m = _vrange_inputs(ch['label'], *_global_ranges[ch['label']], 'mul')
                    multi_entries.append({
                        'label': ch['label'],
                        'color': ch['color'],
                        'matrix': mat,
                        'vmin': vmin_m,
                        'vmax': vmax_m,
                        'sorted_tofs': aligned_tofs,
                    })

                mul_c1, mul_c2 = st.columns([2, 2])
                with mul_c1:
                    mul_bg = st.radio(
                        'Background color', ['White', 'Black'],
                        horizontal=True, key='mul_bg',
                    ).lower()
                with mul_c2:
                    mul_auto, mul_btn = _update_controls('mul_auto', 'mul_btn')

                if multi_entries and (mul_auto or mul_btn):
                    _fig = make_multi_heatmap(multi_entries, prf_well, bg_color=mul_bg)
                    if prf_x_range:
                        _fig.update_xaxes(range=prf_x_range)
                    st.session_state['prf_mul_fig'] = _fig

                cached_mul = st.session_state.get('prf_mul_fig')
                if cached_mul is not None:
                    st.plotly_chart(cached_mul, use_container_width=True, height=800)
                    _image_dl_buttons(cached_mul, f'profile_multi_{prf_well}', 'mul_img')
                else:
                    st.info('Click "Update plot" to render.')

                m1, m2 = st.columns(2)
                m1.metric('Events displayed (common)', len(aligned_ids))
                m2.metric('Total events in well', len(well_ids))

    # ── 3. Overlay Heatmap (additive RGB composite) ───────────────────────────
    with sub_ovhm:
        if len(active_channels) < 2:
            st.info('Upload at least two channel files to use the overlay view.')
        else:
            matrices_ov, ov_ids, ov_tofs = _get_aligned_matrices(max_events)

            if not ov_ids:
                st.warning(f'No events common to all channels for well {prf_well}.')
            else:
                st.markdown(
                    'Channels are blended additively (fluorescence-microscopy style). '
                    'Set each channel\'s range below to control its contribution.'
                )
                ovhm_entries = []
                for ch in active_channels:
                    mat = matrices_ov.get(ch['label'])
                    if mat is None or mat.size == 0:
                        continue
                    vmin_o, vmax_o = _vrange_inputs(ch['label'], *_global_ranges[ch['label']], 'ovh')
                    ovhm_entries.append({
                        'label': ch['label'],
                        'color': ch['color'],
                        'matrix': mat,
                        'vmin': vmin_o,
                        'vmax': vmax_o,
                        'sorted_tofs': ov_tofs,
                    })

                ovh_bg_c, ovh_ctl_c = st.columns([2, 2])
                with ovh_bg_c:
                    ovh_bg = st.radio(
                        'Background color', ['Black', 'White'],
                        horizontal=True, key='ovh_bg',
                        help='Additive blending looks most natural on black.',
                    ).lower()
                with ovh_ctl_c:
                    ovh_auto, ovh_btn = _update_controls('ovh_auto', 'ovh_btn')

                if ovhm_entries and (ovh_auto or ovh_btn):
                    _fig = make_overlay_heatmap(ovhm_entries, prf_well, bg_color=ovh_bg)
                    if prf_x_range:
                        _fig.update_xaxes(range=prf_x_range)
                    st.session_state['prf_ovh_fig'] = _fig

                cached_ovh = st.session_state.get('prf_ovh_fig')
                if cached_ovh is not None:
                    st.plotly_chart(cached_ovh, use_container_width=True, height=800)
                    _image_dl_buttons(cached_ovh, f'profile_overlay_{prf_well}', 'ovh_img')
                else:
                    st.info('Click "Update plot" to render.')

                m1, m2 = st.columns(2)
                m1.metric('Events displayed (common)', len(ov_ids))
                m2.metric('Total events in well', len(well_ids))

    # ── 4. Line Profiles ──────────────────────────────────────────────────────
    with sub_line:
        lp_c1, lp_c2, lp_c3 = st.columns([1, 1, 2])
        with lp_c1:
            lp_ymin = st.number_input('Y-axis min (blank = auto)', value=None,
                                      placeholder='auto', key='prf_lp_ymin')
        with lp_c2:
            lp_ymax = st.number_input('Y-axis max (blank = auto)', value=None,
                                      placeholder='auto', key='prf_lp_ymax')
        with lp_c3:
            lp_norm = st.checkbox(
                'Normalize each channel independently (0–1)',
                value=False, key='lp_norm',
                help='Normalizes each channel to its own 1st–99th percentile range '
                     'so profile shapes can be compared regardless of intensity scale.',
            )

        lp_auto, lp_btn = _update_controls('lp_auto', 'lp_btn')

        lp_entries = []
        for ch in active_channels:
            ch_df = profile_dfs[ch['label']]
            lp_file_meta = f'{ch["file"].name}:{ch["file"].size}'
            lp_mat_key = f'lp_mat|{lp_file_meta}|{prf_well}'
            if lp_mat_key not in st.session_state:
                _lp_mat, _, _ = build_profile_matrix(
                    ch_df, well_ids, tof_series, max_events=len(well_ids)
                )
                st.session_state[lp_mat_key] = _lp_mat
            lp_mat = st.session_state[lp_mat_key]
            if lp_mat.size > 0:
                lp_entries.append({
                    'label': ch['label'],
                    'color': ch['color'],
                    'matrix': lp_mat,
                })

        if not lp_entries:
            st.warning(f'No matching profile data found for well {prf_well}.')
        else:
            if lp_auto or lp_btn:
                _fig = make_overlay_figure(
                    lp_entries, prf_well,
                    y_min=lp_ymin if lp_ymin else None,
                    y_max=lp_ymax if lp_ymax else None,
                    normalize_per_channel=lp_norm,
                )
                if prf_x_range:
                    _fig.update_xaxes(range=prf_x_range)
                st.session_state['prf_lp_fig'] = _fig

            cached_lp = st.session_state.get('prf_lp_fig')
            if cached_lp is not None:
                st.plotly_chart(cached_lp, use_container_width=True, height=500)
                _image_dl_buttons(cached_lp, f'line_profiles_{prf_well}', 'lp_img')
            else:
                st.info('Click "Update plot" to render.')

            lp_rows = {'position': np.arange(lp_entries[0]['matrix'].shape[1])}
            for entry in lp_entries:
                lbl = entry['label']
                mat = entry['matrix']
                n_ev = mat.shape[0]
                lp_rows[f'{lbl}_mean'] = mat.mean(axis=0)
                lp_rows[f'{lbl}_ci95'] = (
                    1.96 * mat.std(axis=0, ddof=1) / np.sqrt(n_ev) if n_ev >= 2 else 0
                )
            lp_export = pd.DataFrame(lp_rows)
            st.download_button(
                '⬇ Download line profile stats CSV',
                data=to_csv_bytes(lp_export),
                file_name=f'line_profiles_{prf_well}.csv',
                mime='text/csv',
                key='prf_dl_lp',
            )


if active_view == 'Worm Profiles':
    _render_profiles_tab()


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown('---')
st.markdown(
    '<div style="text-align:center; color:#888; font-size:0.85rem; padding:8px 0 16px;">'
    'Built by <strong>Kielen Zuurbier, PhD</strong>'
    ' &nbsp;·&nbsp; Built with '
    '<a href="https://claude.ai/code" style="color:#888;">Claude Code</a>'
    ' <span style="font-size:0.8em;">(claude-sonnet-4-6)</span>'
    ' &nbsp;·&nbsp; Powered by '
    '<a href="https://python.org" style="color:#888;">Python</a>, '
    '<a href="https://streamlit.io" style="color:#888;">Streamlit</a>, '
    '<a href="https://plotly.com/python/" style="color:#888;">Plotly</a>, '
    '<a href="https://pandas.pydata.org" style="color:#888;">Pandas</a>, '
    '<a href="https://numpy.org" style="color:#888;">NumPy</a>, '
    '<a href="https://scipy.org" style="color:#888;">SciPy</a>'
    '</div>',
    unsafe_allow_html=True,
)
