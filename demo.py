from __future__ import annotations

from pathlib import Path

import pandas as pd

from preprocessing.feature_extractor import extractFeatures
from preprocessing.windowing import add_time_windows
from graph.graph_builder import build_temporal_graphs
from forecasting import ForecastEngine
from explainability import build_explanation
from policy import RulePolicy
from deceptra import DecoyManager, divert_session, SessionCapture


def run_demo(csv_path: str, window_seconds: int = 5) -> None:
    print(f"[1/7] Loading CSV: {csv_path}")
    raw_df = pd.read_csv(csv_path)

    print("[2/7] Normalizing features")
    canonical_df = extractFeatures(raw_df, source_file=Path(csv_path).name)

    print("[3/7] Adding time windows")
    windowed_df = add_time_windows(canonical_df, window_seconds=window_seconds)

    print("[4/7] Building temporal graphs")
    snapshots = build_temporal_graphs(windowed_df)
    print(f"Built {len(snapshots)} snapshot(s)")

    if not snapshots:
        print("No graphs built. Exiting.")
        return

    print("[5/7] Forecasting")
    forecast_engine = ForecastEngine()
    forecast = forecast_engine.forecast(snapshots)
    print("Forecast:", forecast)

    print("[6/7] Explaining")
    features = windowed_df.iloc[-1].to_dict()
    explanation = build_explanation(
        forecast=forecast,
        graph=snapshots[-1].graph,
        features=features,
        window_summaries=[
            {"attack_probability": forecast.get("attack_probability", 0.0)},
            {"target_score": forecast.get("target_score", 0.0)},
        ],
    )
    print("Summary:", explanation["summary"])

    print("[7/7] Policy + Deceptra")
    policy = RulePolicy("policy/policy_config.yaml")
    asset_criticality = 1
    if "host.asset_criticality" in windowed_df.columns:
        try:
            asset_criticality = int(windowed_df["host.asset_criticality"].fillna(1).iloc[-1])
        except Exception:
            asset_criticality = 1

    decision = policy.decide(forecast, asset_criticality=asset_criticality)
    print("Policy decision:", decision)

    if decision.action == "divert":
        decoy_manager = DecoyManager()
        decoy = decoy_manager.register_decoy(
            decoy_id="server-03-clone",
            role="server",
            subnet="10.0.9.0/24",
            os="Linux",
        )

        diversion = divert_session(
            source=str(windowed_df.iloc[-1].get("flow.src_ip", "unknown")),
            original_target="real-target",
            decoy_target=decoy.decoy_id,
            reason=decision.reason,
        )

        capture = SessionCapture()
        session = capture.start_session("session-1", diversion.source, diversion.decoy_target)
        capture.record_event("session-1", {"event": "diverted"})
        capture.record_event("session-1", {"event": "session_started"})

        print("Diversion:", diversion)
        print("Captured session:", session)
    else:
        print("No diversion triggered.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the NetWorld demo pipeline")
    parser.add_argument("csv_path", help="Path to sample CSV file")
    parser.add_argument("--window-seconds", type=int, default=5)
    args = parser.parse_args()

    run_demo(args.csv_path, window_seconds=args.window_seconds)