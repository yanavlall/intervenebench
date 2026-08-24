"""Portable static explorer for the aggregate-only prospective case study."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Mapping


DEFAULT_RESULTS_EXPLORER_PATH = Path("docs/results/index.html")


def _percent(value: float, maximum: float) -> str:
    if maximum <= 0:
        return "0.0"
    return f"{max(2.0, 100.0 * value / maximum):.1f}"


def _decision_card(label: str, decision: Mapping[str, Any]) -> str:
    passed = decision["decision"] != "hold"
    status = (
        "Limited research use"
        if decision["decision"] == "limited_research_use"
        else "Pass"
        if passed
        else "Hold"
    )
    reasons = "".join(f"<li>{escape(reason)}</li>" for reason in decision["reasons"])
    tone = "pass" if passed else "hold"
    return f"""
      <article class="decision {tone}">
        <div class="decision-top"><span>{escape(label)}</span><span class="pill">{status}</span></div>
        <ul>{reasons}</ul>
      </article>"""


def build_results_explorer_html(report: Mapping[str, Any]) -> str:
    payload = report["payload"]
    decisions = report["release_decisions"]
    scope = payload["evidence_scope"]
    integrity = payload["run_integrity"]
    evidence = payload["decision_evidence"]
    exact = evidence["exact_choice"]
    regret = evidence["normalized_regret"]
    trust = evidence["trust_diagnostics"]
    provenance = payload["provenance"]["source_artifacts"]

    regret_rows = (
        ("Primary simulator", regret["primary_mean"], "primary"),
        ("No-effect control", regret["control_mean"], "reference"),
        ("Uniform action", regret["uniform_mean"], "reference"),
        ("Classical baseline", regret["classical_mean"], "reference"),
    )
    maximum_regret = max(value for _, value, _ in regret_rows)
    bars = "".join(
        f"""
          <div class="bar-row">
            <div class="bar-label"><span>{escape(label)}</span><strong>{value:.4f}</strong></div>
            <div class="track"><div class="bar {tone}" style="width:{_percent(value, maximum_regret)}%"></div></div>
          </div>"""
        for label, value, tone in regret_rows
    )
    decision_cards = "".join(
        (
            _decision_card("Candidate screening", decisions["candidate_screening"]),
            _decision_card(
                "Autonomous intervention selection",
                decisions["autonomous_intervention_selection"],
            ),
            _decision_card(
                "Confidence-based abstention",
                decisions["confidence_based_abstention"],
            ),
            _decision_card(
                "Small-sample human fallback",
                decisions["small_sample_human_fallback"],
            ),
        )
    )
    source_rows = "".join(
        f"<li><code>{escape(source['path'])}</code><span>{escape(source['payload_sha256'][:12])}…</span></li>"
        for source in provenance
    )
    uniform_ci = regret["primary_minus_uniform_confidence_interval"]
    schema_rate = (
        integrity["schema_valid_model_outputs"] / integrity["planned_model_outputs"]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>InterveneBench — Prospective evaluation</title>
  <style>
    :root {{ --ink:#17201d; --muted:#65726d; --paper:#f5f2ea; --card:#fffdf8; --line:#d9d4c8; --green:#176b4a; --green-soft:#dceee4; --amber:#9a5b12; --amber-soft:#f5e6cf; --blue:#315e8a; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--paper); font:16px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ max-width:1120px; margin:auto; padding:64px 28px 80px; }}
    header {{ display:grid; grid-template-columns:1.5fr 1fr; gap:40px; align-items:end; margin-bottom:40px; }}
    .eyebrow {{ color:var(--green); font-size:.78rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }}
    h1 {{ max-width:760px; margin:.3rem 0 1rem; font:700 clamp(2.5rem,6vw,5rem)/.98 ui-serif,Georgia,serif; letter-spacing:-.045em; }}
    .lede {{ max-width:720px; color:var(--muted); font-size:1.12rem; }}
    .stamp {{ border-left:3px solid var(--green); padding:12px 0 12px 20px; color:var(--muted); }}
    .stamp strong {{ display:block; color:var(--ink); font-size:1.1rem; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:28px 0 46px; }}
    .metric,.panel,.decision {{ background:var(--card); border:1px solid var(--line); border-radius:16px; box-shadow:0 8px 28px rgba(37,46,42,.05); }}
    .metric {{ padding:22px; }}
    .metric strong {{ display:block; font:700 2rem/1 ui-serif,Georgia,serif; margin-bottom:8px; }}
    .metric span {{ color:var(--muted); font-size:.9rem; }}
    section {{ margin-top:48px; }}
    h2 {{ margin:0 0 8px; font:700 2rem/1.1 ui-serif,Georgia,serif; letter-spacing:-.025em; }}
    .section-note {{ color:var(--muted); margin:0 0 22px; }}
    .decisions {{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; }}
    .decision {{ padding:20px 22px; border-top:4px solid var(--amber); }}
    .decision.pass {{ border-top-color:var(--green); }}
    .decision-top {{ display:flex; justify-content:space-between; gap:16px; align-items:start; font-weight:750; }}
    .pill {{ white-space:nowrap; border-radius:999px; padding:4px 9px; background:var(--amber-soft); color:var(--amber); font-size:.75rem; text-transform:uppercase; letter-spacing:.04em; }}
    .pass .pill {{ background:var(--green-soft); color:var(--green); }}
    .decision ul {{ color:var(--muted); padding-left:18px; margin:14px 0 0; font-size:.9rem; }}
    .two-col {{ display:grid; grid-template-columns:1.2fr .8fr; gap:18px; }}
    .panel {{ padding:26px; }}
    .bar-row {{ margin:20px 0; }}
    .bar-label {{ display:flex; justify-content:space-between; margin-bottom:7px; font-size:.9rem; }}
    .track {{ height:11px; background:#ece7dc; border-radius:999px; overflow:hidden; }}
    .bar {{ height:100%; border-radius:999px; background:#9aa5a0; }}
    .bar.primary {{ background:var(--green); }}
    .finding {{ display:grid; gap:15px; }}
    .finding div {{ border-bottom:1px solid var(--line); padding-bottom:14px; }}
    .finding div:last-child {{ border:0; padding:0; }}
    .finding strong {{ display:block; font-size:1.35rem; }}
    .finding span {{ color:var(--muted); font-size:.9rem; }}
    .boundary {{ border-left:4px solid var(--blue); }}
    .boundary ul {{ color:var(--muted); }}
    .provenance {{ list-style:none; padding:0; margin:18px 0 0; }}
    .provenance li {{ display:flex; justify-content:space-between; gap:18px; padding:9px 0; border-top:1px solid var(--line); color:var(--muted); font-size:.8rem; }}
    code {{ color:var(--ink); overflow-wrap:anywhere; }}
    footer {{ margin-top:52px; padding-top:22px; border-top:1px solid var(--line); color:var(--muted); font-size:.83rem; }}
    @media (max-width:820px) {{ header,.two-col {{ grid-template-columns:1fr; }} .metrics {{ grid-template-columns:repeat(2,1fr); }} .decisions {{ grid-template-columns:1fr; }} }}
    @media (max-width:480px) {{ main {{ padding:36px 18px 56px; }} .metrics {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <div class="eyebrow">Behavioral simulator evaluation</div>
      <h1>InterveneBench</h1>
      <p class="lede">Can synthetic behavioral evidence choose interventions that real randomized experiments identify as best—and can an evaluator know when not to trust it?</p>
    </div>
    <div class="stamp"><strong>Prospective confirmation</strong>Recommendations and diagnostics were frozen before human outcomes were revealed.</div>
  </header>

  <div class="metrics">
    <div class="metric"><strong>{scope['prospective_confirmation_experiment_count']}</strong><span>prospective experiments</span></div>
    <div class="metric"><strong>{scope['normalized_confirmation_task_count']}</strong><span>normalized decision tasks</span></div>
    <div class="metric"><strong>{exact['count']} / {exact['experiment_count']}</strong><span>exact intervention choices</span></div>
    <div class="metric"><strong>{integrity['schema_valid_model_outputs']} / {integrity['planned_model_outputs']}</strong><span>schema-valid outputs ({schema_rate:.1%})</span></div>
  </div>

  <section>
    <h2>Release decisions, not one flattering score</h2>
    <p class="section-note">Each operational scope has an independent gate. Passing candidate screening does not authorize autonomous use.</p>
    <div class="decisions">{decision_cards}
    </div>
  </section>

  <section>
    <h2>Decision regret</h2>
    <p class="section-note">Lower is better. Regret measures the human utility lost by following the selected intervention instead of the human-best intervention.</p>
    <div class="two-col">
      <article class="panel">{bars}
      </article>
      <article class="panel finding">
        <div><strong>{regret['primary_minus_uniform_mean']:.4f}</strong><span>mean primary-minus-uniform regret</span></div>
        <div><strong>[{uniform_ci[0]:.4f}, {uniform_ci[1]:.4f}]</strong><span>95% experiment-cluster bootstrap interval</span></div>
        <div><strong>{regret['uniform_random_action_tail_probability']:.3f}</strong><span>exact finite-action random tail probability</span></div>
      </article>
    </div>
  </section>

  <section>
    <h2>Failure prediction and fallback</h2>
    <div class="two-col">
      <article class="panel finding">
        <div><strong>{trust['aurc']:.3f} vs {trust['random_abstention_expected_aurc']:.3f}</strong><span>trust AURC versus random abstention; lower is better</span></div>
        <div><strong>{trust['exact_choice_auroc']:.3f}</strong><span>exact-choice trust AUROC</span></div>
        <div><strong>No threshold</strong><span>confidence gate was not validated</span></div>
      </article>
      <article class="panel boundary">
        <h3>What the evidence supports</h3>
        <p>{escape(payload['claim_boundary']['supported'].capitalize())}.</p>
        <h3>What it does not support</h3>
        <ul>{''.join(f'<li>{escape(item)}</li>' for item in payload['claim_boundary']['not_supported'])}</ul>
      </article>
    </div>
  </section>

  <section>
    <h2>Reproducibility boundary</h2>
    <p class="section-note">This page is generated from a self-verifying aggregate-only artifact. It contains no participant rows, experiment-level human scores, human arm means, or human treatment effects.</p>
    <article class="panel">
      <strong>Hash-bound source artifacts</strong>
      <ul class="provenance">{source_rows}</ul>
    </article>
  </section>

  <footer>Small, noncanonical prospective panel. Negative findings are retained: exact choices remain chance-compatible, the trust ranking failed, and tested limited-human fallback did not improve decisions.</footer>
</main>
</body>
</html>
"""
