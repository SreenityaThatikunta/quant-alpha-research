"""Generate committed report figures and a no-dependency research dashboard.

The raw and processed research data stay out of Git.  This script converts a
completed baseline and execution-delay run into compact SVG figures and an HTML
dashboard that can be reviewed directly on GitHub or opened locally.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


NAVY = "#0f172a"
SLATE = "#475569"
GRID = "#cbd5e1"
BLUE = "#2563eb"
ORANGE = "#ea580c"
RED = "#dc2626"
GREEN = "#16a34a"
WIDTH, HEIGHT = 960, 440
MARGIN = {"left": 78, "right": 28, "top": 48, "bottom": 62}


def fmt_percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _scale(value: float, minimum: float, maximum: float, low: float, high: float) -> float:
    if maximum == minimum:
        return (low + high) / 2
    return low + (value - minimum) / (maximum - minimum) * (high - low)


def _line_chart(
    series: list[tuple[str, pd.Series, str]], title: str, y_label: str, output: Path,
    percent_axis: bool = True,
) -> None:
    plot_left, plot_right = MARGIN["left"], WIDTH - MARGIN["right"]
    plot_top, plot_bottom = MARGIN["top"], HEIGHT - MARGIN["bottom"]
    dates = pd.DatetimeIndex(sorted(set().union(*(set(values.index) for _, values, _ in series))))
    date_positions = {pd.Timestamp(date): position for position, date in enumerate(dates)}
    values = [float(item) for _, items, _ in series for item in items.dropna()]
    ymin, ymax = min(values), max(values)
    padding = max((ymax - ymin) * 0.08, 0.002)
    ymin, ymax = ymin - padding, ymax + padding
    x_min, x_max = 0.0, float(max(len(dates) - 1, 1))
    lines: list[str] = []
    for tick in range(5):
        value = ymin + (ymax - ymin) * tick / 4
        y = _scale(value, ymin, ymax, plot_bottom, plot_top)
        label = fmt_percent(value, 0) if percent_axis else f"{value:.2f}"
        lines.append(f'<line x1="{plot_left}" x2="{plot_right}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        lines.append(f'<text x="{plot_left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{label}</text>')
    for position in [round(index * (len(dates) - 1) / 4) for index in range(5)]:
        date = dates[position]
        x = _scale(float(position), x_min, x_max, plot_left, plot_right)
        lines.append(f'<text x="{x:.1f}" y="{plot_bottom + 24}" text-anchor="middle" class="axis">{date:%Y}</text>')
    paths = []
    legend = []
    for index, (label, items, color) in enumerate(series):
        points = []
        for date, value in items.dropna().items():
            x = _scale(float(date_positions[pd.Timestamp(date)]), x_min, x_max, plot_left, plot_right)
            y = _scale(float(value), ymin, ymax, plot_bottom, plot_top)
            points.append(f"{x:.1f},{y:.1f}")
        paths.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        lx = plot_left + index * 170
        legend.append(f'<line x1="{lx}" x2="{lx + 20}" y1="24" y2="24" stroke="{color}" stroke-width="3"/><text x="{lx + 27}" y="28" class="legend">{html.escape(label)}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">
<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(y_label)} across the completed out-of-sample study.</desc>
<style>.axis{{fill:{SLATE};font:12px system-ui,sans-serif}}.legend{{fill:{NAVY};font:13px system-ui,sans-serif}}.title{{fill:{NAVY};font:600 18px system-ui,sans-serif}}.label{{fill:{SLATE};font:13px system-ui,sans-serif}}</style>
<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="white"/>
<text x="{plot_left}" y="24" class="title">{html.escape(title)}</text>{''.join(legend)}
<rect x="{plot_left}" y="{plot_top}" width="{plot_right - plot_left}" height="{plot_bottom - plot_top}" fill="none" stroke="{GRID}"/>
{''.join(lines)}{''.join(paths)}
<text x="18" y="{(plot_top + plot_bottom) / 2:.1f}" class="label" transform="rotate(-90 18 {(plot_top + plot_bottom) / 2:.1f})">{html.escape(y_label)}</text>
</svg>'''
    output.write_text(svg, encoding="utf-8")


def _cost_chart(sensitivity: pd.DataFrame, output: Path) -> None:
    baseline = sensitivity.loc[sensitivity["portfolio_notional"] == sensitivity["portfolio_notional"].min()].sort_values("cost_assumption_bps")
    maximum = max(0.5, float(baseline["sharpe"].abs().max()) + 0.15)
    bar_width, start, baseline_y = 110, 240, 260
    bars = []
    for index, row in enumerate(baseline.itertuples()):
        value = float(row.sharpe)
        height = abs(value) / maximum * 150
        x = start + index * 190
        y = baseline_y - height if value >= 0 else baseline_y
        color = GREEN if value >= 0 else RED
        bars.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{height:.1f}" rx="3" fill="{color}"/>')
        bars.append(f'<text x="{x + bar_width / 2}" y="{baseline_y + 24}" text-anchor="middle" class="axis">{int(row.cost_assumption_bps)} bps</text>')
        bars.append(f'<text x="{x + bar_width / 2}" y="{y - 8 if value >= 0 else y + height + 18:.1f}" text-anchor="middle" class="value">{value:.2f}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">
<title id="title">Liquidity-cost sensitivity</title><desc id="desc">Sharpe ratio for fixed baseline weights at a one-million-dollar notional under increasing half-spread assumptions.</desc>
<style>.axis{{fill:{SLATE};font:13px system-ui,sans-serif}}.value{{fill:{NAVY};font:600 14px system-ui,sans-serif}}.title{{fill:{NAVY};font:600 18px system-ui,sans-serif}}.label{{fill:{SLATE};font:13px system-ui,sans-serif}}</style>
<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="white"/>
<text x="78" y="32" class="title">Liquidity-cost sensitivity — fixed OOS weights, $1m notional</text>
<line x1="120" x2="900" y1="{baseline_y}" y2="{baseline_y}" stroke="{NAVY}" stroke-width="1.5"/>
<text x="108" y="{baseline_y + 4}" text-anchor="end" class="axis">0.00</text>{''.join(bars)}
<text x="510" y="340" text-anchor="middle" class="label">Half-spread assumption</text>
<text x="28" y="220" class="label" transform="rotate(-90 28 220)">Sharpe ratio</text>
</svg>'''
    output.write_text(svg, encoding="utf-8")


def _downsample(backtest: pd.DataFrame) -> list[dict[str, float | str]]:
    sampled = backtest.set_index("date").resample("ME").last().dropna().reset_index()
    return [{"date": row.date.strftime("%Y-%m-%d"), "equity": round(float(row.equity), 5)} for row in sampled.itertuples()]


def _dashboard_html(payload: dict[str, object]) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Alpha Research Dashboard</title>
<style>
:root{{--ink:#0f172a;--muted:#475569;--line:#dbe3ed;--blue:#2563eb;--orange:#ea580c;--green:#16a34a;--red:#dc2626;--bg:#f8fafc;--card:#fff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px system-ui,-apple-system,sans-serif}}main{{max-width:1120px;margin:auto;padding:32px 20px 56px}}h1{{margin:0;font-size:28px}}.sub{{color:var(--muted);margin:8px 0 24px}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}}.label{{color:var(--muted);font-size:12px}}.metric{{font-size:25px;font-weight:650;margin-top:5px}}.layout{{display:grid;grid-template-columns:2fr 1fr;gap:18px;margin-top:18px}}section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px}}h2{{font-size:16px;margin:0 0 12px}}svg{{width:100%;height:auto;display:block}}.note{{color:var(--muted);line-height:1.5}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid var(--line);text-align:right}}td:first-child,th:first-child{{text-align:left}}button{{border:1px solid var(--line);background:#fff;border-radius:6px;padding:7px 10px;margin-right:6px;color:var(--ink);cursor:pointer}}button.active{{background:var(--ink);color:#fff}}@media(max-width:760px){{.grid{{grid-template-columns:repeat(2,1fr)}}.layout{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Point-in-time proxy research dashboard</h1><p class="sub">Nested OOS Ridge study · public-data membership proxy · 2018–2025</p>
<div class="grid" id="metrics"></div>
<div class="layout"><section><h2>Net cumulative return</h2><div><button class="active" data-series="baseline">Baseline entry</button><button data-series="delay">Two-session delay</button></div><svg id="equity" viewBox="0 0 680 320" role="img" aria-label="Net cumulative return chart"></svg></section>
<section><h2>Decision</h2><p class="note" id="decision"></p><h2>Liquidity sensitivity</h2><table><thead><tr><th>Half-spread</th><th>Baseline Sharpe</th><th>Delayed Sharpe</th></tr></thead><tbody id="costs"></tbody></table></section></div>
<section style="margin-top:18px"><h2>Research controls retained in this run</h2><p class="note">Historical membership records are joined only when available by the signal date. Model selection is nested inside expanding historical windows with a five-session embargo. The report keeps fixed-weight cost and capacity sensitivity separate from model selection. The result is rejected because performance decays with execution delay and plausible trading costs.</p></section>
</main><script>const DATA={data};const fmt=v=>(v*100).toFixed(1)+'%';const m=document.getElementById('metrics');DATA.metrics.forEach(x=>m.insertAdjacentHTML('beforeend',`<div class="card"><div class="label">${{x.label}}</div><div class="metric">${{x.value}}</div><div class="label">${{x.note}}</div></div>`));document.getElementById('decision').textContent=DATA.decision;const body=document.getElementById('costs');DATA.costs.forEach(x=>body.insertAdjacentHTML('beforeend',`<tr><td>${{x.bps}} bps</td><td>${{x.baseline}}</td><td>${{x.delay}}</td></tr>`));function draw(key){{const p=DATA[key],svg=document.getElementById('equity'),w=680,h=320,L=58,R=20,T=20,B=42;const vals=p.map(x=>x.equity),min=Math.min(...vals),max=Math.max(...vals),pad=Math.max(.02,(max-min)*.08),lo=min-pad,hi=max+pad;const xy=(d,i)=>[L+i*(w-L-R)/(p.length-1),T+(hi-d.equity)*(h-T-B)/(hi-lo)];const points=p.map(xy).map(a=>a.join(',')).join(' ');let grid='';for(let i=0;i<5;i++){{const v=lo+(hi-lo)*i/4,y=T+(hi-v)*(h-T-B)/(hi-lo);grid+=`<line x1="${{L}}" x2="${{w-R}}" y1="${{y}}" y2="${{y}}" stroke="#dbe3ed"/><text x="${{L-8}}" y="${{y+4}}" text-anchor="end" fill="#475569" font-size="11">${{fmt(v)}}</text>`}}const ticks=[0,Math.floor((p.length-1)/2),p.length-1].map(i=>`<text x="${{xy(p[i],i)[0]}}" y="${{h-14}}" text-anchor="middle" fill="#475569" font-size="11">${{p[i].date.slice(0,4)}}</text>`).join('');svg.innerHTML=grid+`<rect x="${{L}}" y="${{T}}" width="${{w-L-R}}" height="${{h-T-B}}" fill="none" stroke="#cbd5e1"/><polyline points="${{points}}" fill="none" stroke="${{key==='baseline'?'#2563eb':'#ea580c'}}" stroke-width="2.5"/>${{ticks}}`;}}document.querySelectorAll('button[data-series]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('button[data-series]').forEach(x=>x.classList.remove('active'));b.classList.add('active');draw(b.dataset.series)}});draw('baseline');</script></body></html>'''


def _dashboard_markdown(payload: dict[str, object]) -> str:
    metrics = payload["metrics"]
    costs = payload["costs"]
    metric_rows = "\n".join(f"| {item['label']} | {item['value']} | {item['note']} |" for item in metrics)  # type: ignore[index]
    cost_rows = "\n".join(f"| {item['bps']} bps | {item['baseline']} | {item['delay']} |" for item in costs)  # type: ignore[index]
    return f'''# Point-in-time proxy research dashboard

Completed nested out-of-sample Ridge study using a public historical-membership
proxy. This is an educational research artifact, not investment advice.

| Metric | Result | Context |
| --- | ---: | --- |
{metric_rows}

{payload['decision']}

## Net cumulative return

![Net cumulative return comparison](figures/equity_curve.svg)

## Forecast stability

![Rolling rank IC comparison](figures/rank_ic_decay.svg)

## Cost sensitivity

| Half-spread assumption | Baseline Sharpe | Two-session-delay Sharpe |
| --- | ---: | ---: |
{cost_rows}

![Liquidity-cost sensitivity](figures/liquidity_sensitivity.svg)

For an interactive local view, open [dashboard.html](dashboard.html).
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SVG research figures and an HTML dashboard.")
    parser.add_argument("--baseline", type=Path, default=Path("data/processed/sp500_pit_proxy"))
    parser.add_argument("--delay", type=Path, default=Path("data/processed/sp500_pit_proxy_delay2"))
    parser.add_argument("--output", type=Path, default=Path("reports/figures"))
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    baseline, delay = (pd.read_csv(path / "backtest.csv", parse_dates=["date"]) for path in (arguments.baseline, arguments.delay))
    for frame in (baseline, delay):
        frame["equity"] = (1 + frame["net_return"]).cumprod() - 1
    _line_chart([("Baseline entry", baseline.set_index("date")["equity"], BLUE), ("Two-session delay", delay.set_index("date")["equity"], ORANGE)], "Net cumulative return: implementation-delay comparison", "Cumulative return", arguments.output / "equity_curve.svg")

    baseline_ic = pd.read_csv(arguments.baseline / "daily_ic.csv", parse_dates=["date"]).set_index("date")["rank_ic"].rolling(13, min_periods=4).mean()
    delay_ic = pd.read_csv(arguments.delay / "daily_ic.csv", parse_dates=["date"]).set_index("date")["rank_ic"].rolling(13, min_periods=4).mean()
    _line_chart([("Baseline entry", baseline_ic, BLUE), ("Two-session delay", delay_ic, ORANGE)], "Rolling 13-week rank IC", "Rank IC", arguments.output / "rank_ic_decay.svg", percent_axis=False)

    baseline_sensitivity = pd.read_csv(arguments.baseline / "execution_cost_capacity_sensitivity.csv")
    delay_sensitivity = pd.read_csv(arguments.delay / "execution_cost_capacity_sensitivity.csv")
    _cost_chart(baseline_sensitivity, arguments.output / "liquidity_sensitivity.svg")
    minimum_notional = baseline_sensitivity["portfolio_notional"].min()
    costs = []
    for bps in sorted(baseline_sensitivity["cost_assumption_bps"].unique()):
        costs.append({
            "bps": int(bps),
            "baseline": f"{baseline_sensitivity.loc[(baseline_sensitivity.cost_assumption_bps == bps) & (baseline_sensitivity.portfolio_notional == minimum_notional), 'sharpe'].iloc[0]:.2f}",
            "delay": f"{delay_sensitivity.loc[(delay_sensitivity.cost_assumption_bps == bps) & (delay_sensitivity.portfolio_notional == minimum_notional), 'sharpe'].iloc[0]:.2f}",
        })
    baseline_summary = pd.read_csv(arguments.baseline / "summary.csv").set_index("metric")["value"]
    delay_summary = pd.read_csv(arguments.delay / "summary.csv").set_index("metric")["value"]
    coverage = pd.read_csv(arguments.baseline / "universe_price_coverage.csv")
    with_prices = int((~coverage["missing_all_prices"]).sum())
    dashboard = {
        "baseline": _downsample(baseline), "delay": _downsample(delay), "costs": costs,
        "metrics": [
            {"label": "Baseline rank IC", "value": fmt_percent(float(baseline_summary["mean_rank_ic"]), 2), "note": "Mean OOS rank correlation"},
            {"label": "Baseline Sharpe", "value": f"{float(baseline_summary['sharpe']):.2f}", "note": "2 bps half-spread baseline"},
            {"label": "Delayed Sharpe", "value": f"{float(delay_summary['sharpe']):.2f}", "note": "Two-session entry delay"},
            {"label": "Historical members", "value": f"{len(coverage):,}", "note": f"{with_prices:,} with free price history"},
        ],
        "decision": "Rejected: performance is fragile to two-session execution delay and to 10–20 bps half-spread assumptions. This dashboard presents a research control, not an alpha claim.",
    }
    (arguments.output.parent / "dashboard.html").write_text(_dashboard_html(dashboard), encoding="utf-8")
    (arguments.output.parent / "dashboard.md").write_text(_dashboard_markdown(dashboard), encoding="utf-8")
    print(f"Wrote figures to {arguments.output} and dashboards to {arguments.output.parent}")


if __name__ == "__main__":
    main()
