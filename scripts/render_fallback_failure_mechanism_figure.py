from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Mapping

from intervenebench.fallback_failure_mechanism import DEFAULT_MECHANISM_AUDIT_PATH
from intervenebench.protocol import verify_envelope


WIDTH = 1280
HEIGHT = 760
OUTPUT_PATH = Path("docs/reports/figures/fallback_failure_mechanism_v1.svg")


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 18,
    weight: int = 400,
    anchor: str = "start",
    fill: str = "#172033",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{escape(value)}</text>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, **attrs: Any) -> str:
    rendered = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {rendered}/>'


def _heat_color(value: float, limit: float = 0.032) -> str:
    strength = min(abs(value) / limit, 1.0)
    if value > 0:
        start, end = (255, 247, 244), (190, 48, 44)
    elif value < 0:
        start, end = (244, 249, 255), (44, 112, 181)
    else:
        return "#f2f4f7"
    rgb = tuple(round(a + (b - a) * strength) for a, b in zip(start, end))
    return "#" + "".join(f"{component:02x}" for component in rgb)


def render_mechanism_figure(payload: Mapping[str, Any], destination: Path) -> None:
    if payload.get("schema_version") != "intervenebench.fallback_failure_mechanism.v1":
        raise ValueError("unexpected mechanism audit schema")
    curves = payload["figure_data"]["cost_regret_curves"]
    heatmap = payload["figure_data"]["eb_task_budget_heatmap"]
    asymmetry = payload["eb_harm_correction_asymmetry"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _text(50, 48, "Why limited-human fallback failed", size=30, weight=700),
        _text(
            50,
            78,
            "Five prospective confirmation experiments · lower regret is better",
            size=17,
            fill="#536174",
        ),
        _text(55, 128, "A", size=20, weight=700, fill="#41506a"),
        _text(86, 128, "Decision-regret change versus synthetic-only", size=21, weight=650),
        _text(718, 128, "B", size=20, weight=700, fill="#41506a"),
        _text(750, 128, "Where EB fusion helped or hurt", size=21, weight=650),
    ]

    # Panel A: log-spaced budget axis with paired experiment-bootstrap intervals.
    plot_x, plot_y, plot_w, plot_h = 86.0, 168.0, 535.0, 410.0
    budgets = [10, 25, 50, 100, 250]
    x_positions = {budget: plot_x + index * (plot_w / 4) for index, budget in enumerate(budgets)}
    y_min, y_max = -0.006, 0.050

    def y_map(value: float) -> float:
        return plot_y + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    for tick in (-0.005, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05):
        y = y_map(tick)
        color = "#26354d" if tick == 0 else "#e7ebf0"
        width = 2 if tick == 0 else 1
        parts.append(_line(plot_x, y, plot_x + plot_w, y, stroke=color, stroke_width=width))
        parts.append(_text(plot_x - 12, y + 5, f"{tick:+.2f}", size=14, anchor="end", fill="#667386"))
    for budget, x in x_positions.items():
        parts.append(_line(x, plot_y + plot_h, x, plot_y + plot_h + 7, stroke="#7f8998", stroke_width=1))
        parts.append(_text(x, plot_y + plot_h + 28, str(budget), size=15, anchor="middle", fill="#4c596b"))
    parts.append(_text(plot_x + plot_w / 2, plot_y + plot_h + 56, "Total pilot observations", size=16, anchor="middle", fill="#4c596b"))
    parts.append(
        f'<text x="24" y="{plot_y + plot_h / 2:.1f}" transform="rotate(-90 24 {plot_y + plot_h / 2:.1f})" '
        'font-family="Inter,Arial,sans-serif" font-size="16" text-anchor="middle" fill="#4c596b">Δ normalized regret</text>'
    )

    style = {
        "human_only_balanced": ("#b12a32", "Humans only"),
        "synthetic_plus_balanced_fixed10": ("#d28522", "Balanced fixed-10"),
        "synthetic_plus_balanced_eb": ("#2166ac", "Balanced EB"),
        "synthetic_plus_hedged_eb": ("#756bb1", "Hedged EB"),
    }
    curve_map = {curve["policy"]: curve for curve in curves}
    for policy in style:
        color, label = style[policy]
        points = curve_map[policy]["points"]
        coordinates: list[tuple[float, float]] = []
        for point in points:
            x = x_positions[int(point["budget"])]
            y = y_map(float(point["mean_delta_regret"]))
            low, high = [y_map(float(value)) for value in point["confidence_interval"]]
            parts.append(_line(x, high, x, low, stroke=color, stroke_width=1.2, opacity=0.38))
            parts.append(_line(x - 4, high, x + 4, high, stroke=color, stroke_width=1.2, opacity=0.38))
            parts.append(_line(x - 4, low, x + 4, low, stroke=color, stroke_width=1.2, opacity=0.38))
            coordinates.append((x, y))
        path = " ".join(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}" for index, (x, y) in enumerate(coordinates))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')
        for x, y in coordinates:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#ffffff" stroke="{color}" stroke-width="2.5"/>')

    legend_y = 640.0
    for index, policy in enumerate(style):
        color, label = style[policy]
        x = 86 + (index % 2) * 250
        y = legend_y + (index // 2) * 30
        parts.append(_line(x, y, x + 30, y, stroke=color, stroke_width=3))
        parts.append(_text(x + 40, y + 5, label, size=15, fill="#3e4a5c"))

    # Panel B: exact task-by-budget EB deltas.
    heat_x, heat_y = 825.0, 185.0
    cell_w, cell_h = 112.0, 58.0
    for column, budget in enumerate(heatmap["budgets"]):
        parts.append(_text(heat_x + column * cell_w + cell_w / 2, heat_y - 15, str(budget), size=15, weight=600, anchor="middle", fill="#4c596b"))
    for row_index, row in enumerate(heatmap["rows"]):
        y = heat_y + row_index * cell_h
        parts.append(_text(heat_x - 14, y + 36, row["experiment_id"], size=15, anchor="end", fill="#3e4a5c"))
        for column, budget in enumerate(heatmap["budgets"]):
            value = float(row["values"][str(budget)])
            x = heat_x + column * cell_w
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w - 4:.1f}" height="{cell_h - 4:.1f}" rx="5" fill="{_heat_color(value)}" stroke="#ffffff"/>')
            label = "0" if abs(value) < 0.00005 else f"{value:+.3f}"
            parts.append(_text(x + (cell_w - 4) / 2, y + 35, label, size=15, weight=600, anchor="middle", fill="#172033"))
    parts.append(_text(heat_x + 1.5 * cell_w, heat_y + 5 * cell_h + 22, "Total pilot observations", size=15, anchor="middle", fill="#4c596b"))

    box_y = 530.0
    parts.extend(
        [
            f'<rect x="720" y="{box_y}" width="505" height="150" rx="12" fill="#f5f7fa" stroke="#dce2e9"/>',
            _text(746, box_y + 34, "Downside asymmetry", size=19, weight=700),
            _text(
                746,
                box_y + 68,
                f'{asymmetry["worsened_cell_count"]} harmful · {asymmetry["improved_cell_count"]} corrective · {asymmetry["unchanged_cell_count"]} unchanged cells',
                size=16,
                fill="#4c596b",
            ),
            _text(
                746,
                box_y + 107,
                f'{asymmetry["harm_to_correction_magnitude_ratio"]:.1f}×',
                size=34,
                weight=750,
                fill="#a62e35",
            ),
            _text(842, box_y + 103, "larger average harm than correction", size=17, weight=600, fill="#3e4a5c"),
            _text(746, box_y + 133, "Direction remains harmful after dropping any one task.", size=14, fill="#667386"),
        ]
    )

    parts.append(_text(50, 731, "Exploratory aggregate pattern audit; not a causal mechanism estimate. Error bars: paired 95% experiment-bootstrap intervals.", size=14, fill="#6b7687"))
    parts.append("</svg>")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(parts) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = verify_envelope(root / DEFAULT_MECHANISM_AUDIT_PATH)
    destination = root / OUTPUT_PATH
    render_mechanism_figure(payload, destination)
    print(destination)


if __name__ == "__main__":
    main()
