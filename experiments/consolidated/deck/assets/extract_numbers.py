# /// script
# requires-python = ">=3.10"
# dependencies = ["metaflow"]
# ///
"""Dump the exact ConsolidatedFlow numbers the deck figures need to deck_data.json.

Reads the persisted local Metaflow store (no rerun) — the same artifacts ferrum_figs.py
plots. Everything downstream (the SVG generators) reads this JSON so every deck number
traces to run ConsolidatedFlow/<id>, never hand-transcribed.

Run:  uv run --python 3.12 python extract_numbers.py
"""
import json
import os
import pathlib

HERE = pathlib.Path(__file__).parent
FLOW_DIR = (HERE / ".." / ".." / "flow").resolve()

os.environ["METAFLOW_DEFAULT_METADATA"] = "local"
os.environ["METAFLOW_DEFAULT_DATASTORE"] = "local"
os.environ["METAFLOW_DATASTORE_SYSROOT_LOCAL"] = str(FLOW_DIR)
os.chdir(FLOW_DIR)

from metaflow import Flow, namespace  # noqa: E402

K = 6
COND_ORDER = ["log_linear", "monotone_curved", "smooth_nonmono", "sharp_mode", "sharp_off"]
COND_LABEL = {"log_linear": "log-lin", "monotone_curved": "curved", "smooth_nonmono": "smooth",
              "sharp_mode": "sharp", "sharp_off": "sharp-off"}


def main():
    namespace(None)
    run = Flow("ConsolidatedFlow").latest_successful_run
    d = run["aggregate"].task.data
    agg, lifts = d.aggregate_results, d.lift_results

    def prauc(method, cond):
        for r in agg:
            c = r["cell"]
            if c.get("K") == K and c["condition"] == cond and r["method"] == method:
                return round(float(r["prauc_mean"]), 4)
        return None

    # absolute PR-AUC: GRU arms + oracle, and MLP log/ple, across conditions
    gru_encs = ["raw", "log", "ple", "projection", "dense"]
    gru = {COND_LABEL[c]: {e: prauc(f"gru_{e}", c) for e in gru_encs} | {"oracle": prauc("oracle", c)}
           for c in COND_ORDER}
    mlp = {COND_LABEL[c]: {e: prauc(f"mlp_{e}", c) for e in ["log", "ple"]} for c in COND_ORDER}

    def lift(arch, cond, enc):
        for e in lifts:
            c = e["cell"]
            if c["arch"] == arch and c["K"] == K and c["condition"] == cond and c["arm"] == f"{arch}_{enc}":
                return e
        return None

    # crossover: raw_gap of ple vs projection over log, per GRU condition
    crossover = {}
    for c in COND_ORDER:
        row = {}
        for enc in ["ple", "projection"]:
            e = lift("gru", c, enc)
            if e:
                row[enc] = {"raw_gap": round(float(e["raw_gap"]), 4),
                            "sig": bool(e["raw_gap_excludes_zero"])}
        crossover[COND_LABEL[c]] = row

    # add-back illustrative rows (same picks ferrum_figs uses)
    picks = [("raw", "sharp_mode"), ("ple", "sharp_mode"), ("projection", "smooth_nonmono"),
             ("ple", "smooth_nonmono"), ("projection", "monotone_curved"), ("ple", "monotone_curved")]
    addback = []
    for enc, cond in picks:
        e = lift("gru", cond, enc)
        if e is None:
            continue
        addback.append({"label": f"{enc} · {COND_LABEL[cond]}", "enc": enc, "cond": COND_LABEL[cond],
                        "raw_gap": round(float(e["raw_gap"]), 4),
                        "raw_gap_lo": round(float(e["raw_gap_lo"]), 4),
                        "raw_gap_hi": round(float(e["raw_gap_hi"]), 4),
                        "dc_lift": round(float(e["dc_lift"]), 4),
                        "deficit": round(float(e["deficit"]), 4),
                        "raw_gap_sig": bool(e["raw_gap_excludes_zero"])})

    out = {"run_id": run.id, "K": K, "gru_prauc": gru, "mlp_prauc": mlp,
           "crossover_raw_gap": crossover, "addback": addback}
    (HERE / "deck_data.json").write_text(json.dumps(out, indent=2))
    print(f"wrote deck_data.json from ConsolidatedFlow/{run.id}")
    print("MLP sharp:", mlp["sharp"], "| GRU sharp:", gru["sharp"])


if __name__ == "__main__":
    main()
