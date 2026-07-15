# /// script
# requires-python = ">=3.10"
# dependencies = ["ferrum-viz>=0.16", "polars", "metaflow"]
# ///
"""Figures for the consolidated encoding-capacity report, drawn with ferrum-viz.

Reads the persisted ConsolidatedFlow artifact store (no rerun): `aggregate` step holds
`aggregate_results` (absolute PR-AUC per cell/arm) and `lift_results` (dc_lift + raw_gap + CIs).

Outputs (this folder):
  fig1_gru_crossover_prauc.png  -- GRU absolute PR-AUC by encoder x risk-shape (the crossover + oracle ceiling)
  fig2_mlp_refutation.png       -- free-nonlinearity MLP: log vs ple by shape (redundant except SHARP)
  fig3_rawgap_vs_dc.png         -- deployment raw_gap (with CI) vs deficit-corrected dc_lift (the add-back)

Run:  uv run ferrum_figs.py
"""
import os
import pathlib

HERE = pathlib.Path(__file__).parent
FLOW_DIR = (HERE / "flow").resolve()

# point the Metaflow client at the local store (SYSROOT_LOCAL = parent of `.metaflow`)
os.environ["METAFLOW_DEFAULT_METADATA"] = "local"
os.environ["METAFLOW_DEFAULT_DATASTORE"] = "local"
os.environ["METAFLOW_DATASTORE_SYSROOT_LOCAL"] = str(FLOW_DIR)
os.chdir(FLOW_DIR)

from metaflow import Flow, namespace  # noqa: E402
import ferrum as fm  # noqa: E402
import polars as pl  # noqa: E402

K = 6
COND_ORDER = ["log_linear", "monotone_curved", "smooth_nonmono", "sharp_mode", "sharp_off"]
COND_LABEL = {"log_linear": "log-lin", "monotone_curved": "curved", "smooth_nonmono": "smooth",
              "sharp_mode": "sharp", "sharp_off": "sharp-off"}
COND_LABELS = [COND_LABEL[c] for c in COND_ORDER]

# encoder arms (bare enc name) + colors; oracle = light-grey ceiling
ENC_ORDER = ["raw", "log", "ple", "projection", "dense", "oracle"]
ENC_COLORS = ["#9e9e9e", "#4c78a8", "#d62728", "#2ca02c", "#9467bd", "#d9d9d9"]
# one color scale per chart (ferrum): fold raw_gap-significance AND the dc_lift point into one "kind" field
KIND_DOMAIN = ["real deployment gap (CI>0)", "gap not sig (CI∋0)", "dc_lift (inflated by add-back)"]
KIND_RANGE = ["#2ca02c", "#bbbbbb", "#1f77b4"]


def load():
    namespace(None)
    run = Flow("ConsolidatedFlow").latest_successful_run
    d = run["aggregate"].task.data
    return run, d.aggregate_results, d.lift_results


def prauc_df(aggregate_results, arch):
    """absolute PR-AUC for one architecture's arms (+ oracle) at K, per condition."""
    rows = []
    for r in aggregate_results:
        c = r["cell"]
        if c.get("K") != K:
            continue
        m = r["method"]
        if m == "oracle":
            enc = "oracle"
        elif m.startswith(arch + "_"):
            enc = m.split("_", 1)[1]
        else:
            continue
        if enc not in ENC_ORDER:
            continue
        rows.append({"condition": COND_LABEL[c["condition"]], "enc": enc,
                     "pr_auc": float(r["prauc_mean"])})
    return (pl.DataFrame(rows)
            .with_columns(pl.col("condition").cast(pl.Enum(COND_LABELS)),
                          pl.col("enc").cast(pl.Enum(ENC_ORDER)))
            .sort("condition", "enc"))


def mlp_df(aggregate_results):
    rows = []
    for r in aggregate_results:
        c = r["cell"]
        if c.get("K") != K or r["method"] not in ("mlp_log", "mlp_ple"):
            continue
        rows.append({"condition": COND_LABEL[c["condition"]],
                     "arm": r["method"].split("_", 1)[1], "pr_auc": float(r["prauc_mean"])})
    return (pl.DataFrame(rows)
            .with_columns(pl.col("condition").cast(pl.Enum(COND_LABELS)),
                          pl.col("arm").cast(pl.Enum(["log", "ple"])))
            .sort("condition", "arm"))


def _lift(lift_results, arch, cond, arm_enc):
    for e in lift_results:
        c = e["cell"]
        if c["arch"] == arch and c["K"] == K and c["condition"] == cond and c["arm"] == f"{arch}_{arm_enc}":
            return e
    return None


def addback_df(lift_results):
    """illustrative gru rows: raw_gap (deployment, with CI) vs dc_lift (structural) — the add-back gap."""
    picks = [("raw", "sharp_mode"), ("ple", "sharp_mode"), ("projection", "smooth_nonmono"),
             ("ple", "smooth_nonmono"), ("projection", "monotone_curved"), ("ple", "monotone_curved")]
    ci, pts = [], []
    for enc, cond in picks:
        e = _lift(lift_results, "gru", cond, enc)
        if e is None:
            continue
        label = f"{enc} · {COND_LABEL[cond]}"
        status = KIND_DOMAIN[0] if e.get("raw_gap_excludes_zero") else KIND_DOMAIN[1]
        ci.append({"label": label, "lo": float(e["raw_gap_lo"]), "hi": float(e["raw_gap_hi"]),
                   "raw_gap": float(e["raw_gap"]), "kind": status})
        pts.append({"label": label, "value": float(e["raw_gap"]), "kind": status})          # raw_gap point
        pts.append({"label": label, "value": float(e["dc_lift"]), "kind": KIND_DOMAIN[2]})   # dc_lift point
    order = [r["label"] for r in sorted(ci, key=lambda r: r["raw_gap"])]
    enum = pl.Enum(order)
    ci_df = pl.DataFrame(ci).with_columns(pl.col("label").cast(enum), pl.col("kind").cast(pl.Enum(KIND_DOMAIN)))
    pts_df = pl.DataFrame(pts).with_columns(pl.col("label").cast(enum), pl.col("kind").cast(pl.Enum(KIND_DOMAIN)))
    return ci_df, pts_df


def fig_bars(df, color_field, domain, colors, title, caption, out, subtitle=""):
    chart = (
        fm.Chart(df).mark_bar(position=fm.Dodge())
        .encode(x="condition:N", y=fm.Y("pr_auc", scale=fm.LinearScale(domain=[0.0, 0.66])),
                color=fm.Color(color_field, scale=fm.OrdinalScale(domain=domain, range=colors)))
        .labs(x="risk shape (of latent log-odds vs value)", y="PR-AUC (8-seed mean)", title=title)
        .properties(width=880, height=470, subtitle=subtitle, caption=caption)
        .theme(fm.themes.publication)
    )
    chart.save(str(out))


def fig_addback(ci_df, pts_df, out):
    color = fm.Color("kind:N", scale=fm.OrdinalScale(domain=KIND_DOMAIN, range=KIND_RANGE))
    rule = fm.Chart(ci_df).mark_rule().encode(y="label:N", x="lo:Q", x2="hi:Q", color=color)
    pts = fm.Chart(pts_df).mark_point().encode(y="label:N", x="value:Q", color=color)
    chart = (
        (rule + pts + fm.annotate_vline(0.0, stroke="black"))
        .labs(x="lift over log (PR-AUC) — coloured CI = raw_gap; blue point = dc_lift; their gap = deficit add-back",
              y="", title="Read the deployment gap, not the estimand: raw_gap vs dc_lift (GRU, K=6)")
        .properties(width=880, height=430,
                    caption="green = real deployment gap; grey = not sig; blue = the inflated dc_lift. "
                            "'raw · sharp': dc_lift +0.16 but raw_gap ≈ 0 (pure add-back).")
        .theme(fm.themes.publication)
    )
    chart.save(str(out))


def main():
    run, agg, lifts = load()
    print(f"loaded ConsolidatedFlow/{run.id} (successful={run.successful}) | "
          f"{len(agg)} agg cells, {len(lifts)} lift rows")

    gdf = prauc_df(agg, "gru")
    fig_bars(gdf, "enc", ENC_ORDER, ENC_COLORS,
             "Affine-input GRU: the encoder crosses over by risk-shape",
             "sharp → fixed PLE (red) wins; smooth/curved → learned projection (green) wins; oracle = grey ceiling",
             HERE / "fig1_gru_crossover_prauc.png",
             subtitle="absolute PR-AUC, K=6")
    print("  fig1 wrote")

    mdf = mlp_df(agg)
    fig_bars(mdf, "arm", ["log", "ple"], ["#4c78a8", "#d62728"],
             "Free-nonlinearity MLP: PLE is redundant — EXCEPT on the sharp band",
             "the refutation: a free per-step nonlinearity does NOT make encoding redundant for a localized target",
             HERE / "fig2_mlp_refutation.png",
             subtitle="MLP log vs ple, absolute PR-AUC, K=6")
    print("  fig2 wrote")

    ci_df, pts_df = addback_df(lifts)
    fig_addback(ci_df, pts_df, HERE / "fig3_rawgap_vs_dc.png")
    print("  fig3 wrote")

    # sanity echo
    print("GRU sharp PR-AUC:", {r["enc"]: r["pr_auc"] for r in
          gdf.filter(pl.col("condition") == "sharp").to_dicts()})


if __name__ == "__main__":
    main()
