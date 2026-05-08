import numpy as np
import pandas as pd
import plotly.graph_objects as go

COLOR_CONTROL = '#e07b54'
COLOR_CONTROL_BORDER = '#c0522a'
COLOR_DEFAULT = '#4c8bbe'
COLOR_DEFAULT_BORDER = '#2e6494'


def _well_colors(wells: list, control_wells: 'str | list[str]'):
    if isinstance(control_wells, str):
        control_wells = [control_wells]
    ctrl_set = set(control_wells)
    fills   = [COLOR_CONTROL        if w in ctrl_set else COLOR_DEFAULT        for w in wells]
    borders = [COLOR_CONTROL_BORDER if w in ctrl_set else COLOR_DEFAULT_BORDER for w in wells]
    return fills, borders


def _base_layout(title: str, y_label: str) -> dict:
    return dict(
        title_text=title,
        title_font=dict(size=20, color='#111'),
        # x-axis title rendered as an annotation by _add_n_annotations so that
        # the tick-label order is: well → n= → "Source Well" (top to bottom)
        xaxis_title='',
        yaxis_title=y_label,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=14, color='#111'),
            title_font=dict(size=16, color='#111'),
            showline=True,
            linewidth=2,
            linecolor='#333',
            ticks='outside',
            ticklen=6,
            gridcolor='#e0e0e0',
        ),
        yaxis=dict(
            tickfont=dict(size=14, color='#111'),
            title_font=dict(size=16, color='#111'),
            showline=True,
            linewidth=2,
            linecolor='#333',
            ticks='outside',
            ticklen=6,
            gridcolor='#e0e0e0',
            zeroline=True,
            zerolinecolor='#aaa',
            zerolinewidth=1.5,
        ),
        height=600,
        margin=dict(t=70, b=155, l=90, r=30),
        font=dict(size=14, color='#111'),
    )


def _add_n_annotations(fig, wells: list, ns: list):
    """Add n=X labels then a 'Source Well' title below the x-axis tick labels."""
    for well, n in zip(wells, ns):
        fig.add_annotation(
            x=well,
            y=-0.11,
            xref='x',
            yref='paper',
            text=f'n={int(n)}',
            showarrow=False,
            font=dict(size=12, color='#444'),
            xanchor='center',
            yanchor='top',
        )
    # "Source Well" axis title sits below all n= labels
    fig.add_annotation(
        x=0.5,
        y=-0.22,
        xref='paper',
        yref='paper',
        text='Source Well',
        showarrow=False,
        font=dict(size=16, color='#111'),
        xanchor='center',
        yanchor='top',
    )


def _add_legend_annotation(fig, color: str, plural: bool = False):
    label = 'Control wells' if plural else 'Control well'
    fig.add_annotation(
        text=f'<span style="color:{color}">■</span> {label}',
        xref='paper', yref='paper',
        x=1.0, y=1.04,
        showarrow=False,
        font=dict(size=13, color='#111'),
        xanchor='right',
    )


def make_bar_figure(
    stats_df: pd.DataFrame,
    control_wells: 'str | list[str]',
    title: str,
    y_label: str,
) -> go.Figure:
    wells = stats_df['well'].tolist()
    ns = stats_df['n'].tolist()
    fills, borders = _well_colors(wells, control_wells)
    ci = stats_df['ci95'].fillna(0).tolist()

    trace = go.Bar(
        x=wells,
        y=stats_df['mean'],
        error_y=dict(
            type='data',
            array=ci,
            visible=True,
            color='#222',
            thickness=2,
            width=5,
        ),
        marker_color=fills,
        marker_line_color=borders,
        marker_line_width=1.5,
        hovertemplate=(
            '<b>%{x}</b><br>'
            'Mean: %{y:.4f}<br>'
            '95% CI: ±%{error_y.array:.4f}<extra></extra>'
        ),
    )

    fig = go.Figure(data=[trace])
    fig.update_layout(**_base_layout(title, y_label))
    _add_n_annotations(fig, wells, ns)
    plural = isinstance(control_wells, list) and len(control_wells) > 1
    _add_legend_annotation(fig, COLOR_CONTROL, plural=plural)
    return fig


def make_violin_figure(
    df: pd.DataFrame,
    param: str,
    wells: list,
    control_wells: 'str | list[str]',
    title: str,
    y_label: str,
) -> go.Figure:
    fills, borders = _well_colors(wells, control_wells)
    traces = []
    ns = []

    for well, fill, border in zip(wells, fills, borders):
        vals = df.loc[df['Source well'] == well, param].dropna().to_numpy(dtype=float)
        n = len(vals)
        ns.append(n)
        if n == 0:
            continue
        traces.append(go.Violin(
            x=[well] * n,
            y=vals,
            name=well,
            fillcolor=fill,
            line_color=border,
            box_visible=True,
            meanline_visible=True,
            points=False,
            showlegend=False,
            hovertemplate=f'<b>{well}</b><br>Value: %{{y:.4f}}<br>n={n}<extra></extra>',
        ))

    fig = go.Figure(data=traces)
    layout = _base_layout(title, y_label)
    layout['violinmode'] = 'overlay'
    fig.update_layout(**layout)
    _add_n_annotations(fig, wells, ns)
    plural = isinstance(control_wells, list) and len(control_wells) > 1
    _add_legend_annotation(fig, COLOR_CONTROL, plural=plural)
    return fig


def make_strip_figure(
    df: pd.DataFrame,
    param: str,
    wells: list,
    control_wells: 'str | list[str]',
    title: str,
    y_label: str,
    max_points: int = 500,
) -> go.Figure:
    rng = np.random.default_rng(seed=42)
    fills, borders = _well_colors(wells, control_wells)
    traces = []
    ns = []

    for well, fill, border in zip(wells, fills, borders):
        vals = df.loc[df['Source well'] == well, param].dropna().to_numpy(dtype=float)
        n_total = len(vals)
        ns.append(n_total)
        if n_total == 0:
            continue

        if n_total > max_points:
            sampled = rng.choice(vals, size=max_points, replace=False)
            hover_note = f'sampled {max_points}/{n_total}'
        else:
            sampled = vals
            hover_note = f'n={n_total}'

        traces.append(go.Box(
            x=[well] * len(sampled),
            y=sampled,
            name=well,
            boxpoints='all',
            jitter=0.4,
            pointpos=0,
            marker=dict(color=fill, size=4, opacity=0.6, line=dict(color=border, width=0.5)),
            line=dict(color='rgba(0,0,0,0)'),
            fillcolor='rgba(0,0,0,0)',
            whiskerwidth=0,
            showlegend=False,
            hovertemplate=f'<b>{well}</b><br>Value: %{{y:.4f}}<br>{hover_note}<extra></extra>',
        ))

    fig = go.Figure(data=traces)
    layout = _base_layout(title, y_label)
    layout['boxmode'] = 'overlay'
    fig.update_layout(**layout)
    _add_n_annotations(fig, wells, ns)
    plural = isinstance(control_wells, list) and len(control_wells) > 1
    _add_legend_annotation(fig, COLOR_CONTROL, plural=plural)
    return fig
