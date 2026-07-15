# /// script
# requires-python = ">=3.10"
# dependencies = ["ferrum-viz>=0.16", "polars", "metaflow"]
# ///
"""Recreate the cycle-6 (affine-input GRU) report figures with ferrum-viz.

Data source: the Metaflow artifact store of the promoted PerStepFlow (its
`aggregate` step holds `aggregate_results` / `lift_results` / `equiv_margin`).
No experiment is rerun — this only reads persisted run artifacts.

Outputs (new files, the matplotlib originals are left untouched):
  figB_affine_gru_prauc_ferrum.png   <- mirrors fig_summary_prauc.png
  figB_affine_gru_lifts_ferrum.png   <- mirrors fig_lift_forest.png

Run:  uv run ferrum_figs.py
"""
import os
import pathlib

HERE = pathlib.Path(__file__).parent
FLOW_DIR = (HERE / ".." / "gru_perstep_encoding_fraud" / "flow").resolve()

# --- point the Metaflow client at the local artifact store -------------------
# SYSROOT_LOCAL is the *parent* of `.metaflow`; metaflow appends `.metaflow`.
os.environ["METAFLOW_DEFAULT_METADATA"] = "local"
os.environ["METAFLOW_DEFAULT_DATASTORE"] = "local"
os.environ["METAFLOW_DATASTORE_SYSROOT_LOCAL"] = str(FLOW_DIR)
os.chdir(FLOW_DIR)

from metaflow import Flow, namespace  # noqa: E402

import ferrum as fm  # noqa: E402
import polars as pl  # noqa: E402

ARMS = ["tab_logreg", "raw", "scalar", "ple", "dense", "oracle"]
GREEN, RED, GREY = "#2ca02c", "#d62728", "#888888"
STATUS_DOMAIN = ["positive (CI sig)", "negative (CI sig)", "overlaps 0"]
STATUS_RANGE = [GREEN, RED, GREY]


def load_run():
    """Read the latest successful PerStepFlow run's aggregate artifacts."""
    namespace(None)  # runs are namespaced by creating user; read globally
    run = Flow("PerStepFlow").latest_run
    agg = run["aggregate"].task.data
    return run, agg.aggregate_results, agg.lift_results, agg.equiv_margin


def prauc_frame(aggregate_results):
    rows = []
    for r in aggregate_results:
        c = r["cell"]
        if r["method"] not in ARMS:
            continue
        rows.append(
            {
                "arm": r["method"],
                "regime": c["regime"],
                "length": f"L={c['length']}",
                "pr_auc": float(r["pr_auc_mean"]),
            }
        )
    df = pl.DataFrame(rows)
    # pin categorical order (grammar libs do not guarantee it otherwise)
    return df.with_columns(
        pl.col("arm").cast(pl.Enum(ARMS)),
        pl.col("length").cast(pl.Enum(["L=32", "L=300"])),
        pl.col("regime").cast(pl.Enum(["band", "monotone"])),
    ).sort("regime", "arm", "length")


def lift_frame(lift_results):
    rows = []
    for e in lift_results:
        c = e["cell"]
        m = float(e["lift_mean"])
        if e.get("ci_excludes_zero"):
            status = STATUS_DOMAIN[0] if m > 0 else STATUS_DOMAIN[1]
        else:
            status = STATUS_DOMAIN[2]
        flag = ("  [Holm]" if e.get("holm_significant") else "") + (
            "  ≡" if e.get("equivalent_to_scalar") else ""
        )
        label = f"{c['regime']} L{c['length']}: {c['pair'].replace('_minus_', '−')}{flag}"
        rows.append(
            {
                "label": label,
                "regime": c["regime"],
                "length": int(c["length"]),
                "lift_mean": m,
                "lift_lo": float(e["lift_lo"]),
                "lift_hi": float(e["lift_hi"]),
                "status": status,
            }
        )
    df = pl.DataFrame(rows).sort("regime", "length", "lift_mean")
    order = df["label"].to_list()  # top-to-bottom y order
    return df.with_columns(
        pl.col("label").cast(pl.Enum(order)),
        pl.col("status").cast(pl.Enum(STATUS_DOMAIN)),
    )


def fig_prauc(df, out):
    # Concat (not facet): lets the right panel drop its y-axis so the inner
    # axis never overlaps the band panel's bars, and gives a single shared legend.
    ymax = float(df["pr_auc"].max())
    yscale = fm.LinearScale(domain=[0.0, ymax + 0.06])
    cscale = fm.OrdinalScale(domain=["L=32", "L=300"], range=["#4c78a8", "#f58518"])

    def panel(regime, *, show_y, show_legend):
        ykw = {"scale": yscale}
        if not show_y:
            ykw["axis"] = None  # hide the inner (right) y-axis
        ckw = {"scale": cscale}
        if not show_legend:
            ckw["legend"] = None  # keep a single legend on the right panel
        return (
            fm.Chart(df.filter(pl.col("regime") == regime))
            .mark_bar(position=fm.Dodge())
            .encode(x="arm:N", y=fm.Y("pr_auc", **ykw), color=fm.Color("length", **ckw))
            .labs(title=regime, x="arm", y=("PR-AUC (5-seed mean)" if show_y else ""))
            .properties(width=560, height=440)
        )

    chart = (
        (panel("band", show_y=True, show_legend=False)
         | panel("monotone", show_y=False, show_legend=True))
        .properties(
            title="Affine-input GRU: PR-AUC by arm and length",
            caption="band: ple/dense beat scalar (unbottlenecking helps); monotone: scalar suffices",
        )
        .theme(fm.themes.publication)
        # NOTE: ferrum 0.16 renders concat-level title/caption flush to the left
        # edge with no spacing, and neither configure_padding nor
        # configure_title(anchor=) overrides it. Tracked upstream:
        # https://github.com/chris-santiago/ferrum/issues/1
    )
    chart.save(str(out))


def fig_lifts(df, equiv_margin, out):
    color = fm.Color(
        "status:N", scale=fm.OrdinalScale(domain=STATUS_DOMAIN, range=STATUS_RANGE)
    )
    rule = fm.Chart(df).mark_rule().encode(y="label:N", x="lift_lo:Q", x2="lift_hi:Q", color=color)
    pts = fm.Chart(df).mark_point().encode(y="label:N", x="lift_mean:Q", color=color)
    chart = (
        rule
        + pts
        + fm.annotate_vline(0.0, stroke="black")
        + fm.annotate_vline(equiv_margin, stroke=GREY, stroke_dash=[4, 3])
        + fm.annotate_vline(-equiv_margin, stroke=GREY, stroke_dash=[4, 3])
    )
    chart = (
        chart.labs(
            x="PR-AUC lift (seed-level paired-t 95% CI; grey band = +/- equivalence margin)",
            y="",
            title="Per-step encoding lifts (seed-level paired): band unbottlenecking vs monotone control",
        )
        .theme(fm.themes.publication)
        .properties(width=760, height=460)
    )
    chart.save(str(out))


def main():
    run, aggregate_results, lift_results, equiv_margin = load_run()
    print(f"loaded PerStepFlow/{run.id} (successful={run.successful})")
    print(f"  {len(aggregate_results)} aggregate cells, {len(lift_results)} lift rows, "
          f"equiv_margin={equiv_margin}")

    pf = prauc_frame(aggregate_results)
    lf = lift_frame(lift_results)

    out1 = HERE / "figB_affine_gru_prauc_ferrum.png"
    out2 = HERE / "figB_affine_gru_lifts_ferrum.png"
    fig_prauc(pf, out1)
    fig_lifts(lf, equiv_margin, out2)
    print("wrote", out1.name, "and", out2.name)

    # sanity echo against CONCLUSIONS (band: ple 0.591/0.579, scalar 0.401/0.372)
    print("band PR-AUC check:")
    for arm in ("scalar", "ple", "dense"):
        vals = {r["length"]: r["pr_auc"] for r in pf.filter(
            (pl.col("regime") == "band") & (pl.col("arm") == arm)).to_dicts()}
        print(f"  {arm:7s} {vals}")


if __name__ == "__main__":
    main()
