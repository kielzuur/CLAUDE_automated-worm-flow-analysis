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
            seq_ctrl1 = st.multiselect(
                'Control well(s)',
                all_wells,
                default=[all_wells[0]] if all_wells else [],
                key='seq_ctrl1',
            )
            seq_op_label1 = st.selectbox('Operation', list(OPERATIONS.keys()), key='seq_op1')
            seq_op_key1 = OPERATIONS[seq_op_label1]

            seq_map1 = {}
            with st.expander('Advanced: per-well mapping (optional)'):
                st.caption('Override the global control for specific target wells.')
                n_seq1 = st.number_input(
                    'Number of groups', min_value=0, max_value=20,
                    value=0, step=1, key='seq1_n_grp',
                )
                seq_map1_targets = []
                _s1_grp_defs = []
                for g in range(int(n_seq1)):
                    st.markdown(f'**Group {g + 1}**')
                    s1gc = st.multiselect('Control well(s)', all_wells, key=f'seq1_grp_ctrl_{g}')
                    s1gt = st.multiselect('Target well(s)', all_wells, key=f'seq1_grp_tgt_{g}')
                    if s1gc and s1gt:
                        for tw in s1gt:
                            seq_map1[tw] = s1gc
                        seq_map1_targets.extend(s1gt)
                        _s1_grp_defs.append(s1gc)

                for _gc in _s1_grp_defs:
                    for cw in _gc:
                        if cw not in seq_map1:
                            seq_map1[cw] = _gc

                seen1: dict = {}
                for tw in seq_map1_targets:
                    seen1[tw] = seen1.get(tw, 0) + 1
                dupes1 = [w for w, c in seen1.items() if c > 1]
                if dupes1:
                    st.warning(f'Duplicate targets (last wins): {", ".join(dupes1)}')

        with sc2:
            st.markdown('### Step 2')
            seq_param2 = st.selectbox(
                'Reference parameter',
                augmented_cols,
                key='seq_param2',
                help='Parameter used to compute the Step 2 reference value from the raw data.',
            )
            seq_ctrl2 = st.multiselect(
                'Control well(s)',
                all_wells,
                default=[all_wells[0]] if all_wells else [],
                key='seq_ctrl2',
            )
            seq_op_label2 = st.selectbox('Operation', list(OPERATIONS.keys()), key='seq_op2')
            seq_op_key2 = OPERATIONS[seq_op_label2]

            seq_map2 = {}
            with st.expander('Advanced: per-well mapping (optional)'):
                st.caption('Override the global reference well for specific target wells.')
                n_seq2 = st.number_input(
                    'Number of groups', min_value=0, max_value=20,
                    value=0, step=1, key='seq2_n_grp',
                )
                seq_map2_targets = []
                _s2_grp_defs = []
                for g in range(int(n_seq2)):
                    st.markdown(f'**Group {g + 1}**')
                    s2gc = st.multiselect('Reference well(s)', all_wells, key=f'seq2_grp_ctrl_{g}')
                    s2gt = st.multiselect('Target well(s)', all_wells, key=f'seq2_grp_tgt_{g}')
                    if s2gc and s2gt:
                        for tw in s2gt:
                            seq_map2[tw] = s2gc
                        seq_map2_targets.extend(s2gt)
                        _s2_grp_defs.append(s2gc)

                for _gc in _s2_grp_defs:
                    for cw in _gc:
                        if cw not in seq_map2:
                            seq_map2[cw] = _gc

                seen2: dict = {}
                for tw in seq_map2_targets:
                    seen2[tw] = seen2.get(tw, 0) + 1
                dupes2 = [w for w, c in seen2.items() if c > 1]
                if dupes2:
                    st.warning(f'Duplicate targets (last wins): {", ".join(dupes2)}')

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
