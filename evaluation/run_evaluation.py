"""Main evaluation script — runs all systems and computes metrics.

Usage:
    python evaluation/run_evaluation.py --config configs/config.yaml
    python evaluation/run_evaluation.py --config configs/config.yaml --system g2cp
    python evaluation/run_evaluation.py --config configs/config.yaml --system ftma --n 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from evaluation.metrics import aggregate_metrics, compute_all_metrics
from g2cp.baselines.ftma import FTMASystem
from g2cp.baselines.jsma import JSMASystem
from g2cp.baselines.single_agent import SingleAgentSystem
from g2cp.main import build_system, load_config
from g2cp.utils.graph_db import GraphDB
from g2cp.utils.llm import LLMClient
from g2cp.utils.tokens import TokenCounter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("evaluation")


def run_g2cp(config: dict, queries: list[dict]) -> list[dict]:
    """Run G²CP system on all queries."""
    runtime = build_system(config)
    results = []

    for i, q in enumerate(queries):
        logger.info(f"[G²CP] Query {i+1}/{len(queries)}")
        result = runtime.process_query(q["query"])
        audit = result.get("audit_log")
        trace = audit.get_trace() if audit else []

        metrics = compute_all_metrics(
            response=result["response"],
            ground_truth=q.get("ground_truth", {}),
            token_count=result["metrics"]["total_tokens"],
            message_trace=trace,
        )
        results.append({
            "query_id": q.get("id"),
            "category": q.get("category"),
            "query": q["query"],
            "response": result["response"],
            "metrics": metrics,
            "raw_metrics": result["metrics"],
        })

    return results


def run_ftma(config: dict, queries: list[dict]) -> list[dict]:
    """Run FTMA baseline on all queries."""
    neo4j_cfg = config.get("neo4j", {})
    db = GraphDB(
        uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
        user=neo4j_cfg.get("user", "neo4j"),
        password=neo4j_cfg.get("password", ""),
    )
    llm_cfg = config.get("llm", {})
    llm = LLMClient(api_key=llm_cfg.get("api_key", ""), model=llm_cfg.get("model", "gpt-4"))

    system = FTMASystem(db=db, llm=llm)
    results = []

    for i, q in enumerate(queries):
        logger.info(f"[FTMA] Query {i+1}/{len(queries)}")
        result = system.process_query(q["query"])

        # FTMA has no structured trace — lower auditability
        metrics = compute_all_metrics(
            response=result["response"],
            ground_truth=q.get("ground_truth", {}),
            token_count=result["metrics"]["total_tokens"],
            message_trace=[],  # No structured trace
        )
        results.append({
            "query_id": q.get("id"),
            "category": q.get("category"),
            "query": q["query"],
            "response": result["response"],
            "metrics": metrics,
            "raw_metrics": result["metrics"],
        })

    return results


def run_jsma(config: dict, queries: list[dict]) -> list[dict]:
    """Run JSMA baseline on all queries."""
    neo4j_cfg = config.get("neo4j", {})
    db = GraphDB(
        uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
        user=neo4j_cfg.get("user", "neo4j"),
        password=neo4j_cfg.get("password", ""),
    )
    llm_cfg = config.get("llm", {})
    llm = LLMClient(api_key=llm_cfg.get("api_key", ""), model=llm_cfg.get("model", "gpt-4"))

    system = JSMASystem(db=db, llm=llm)
    results = []

    for i, q in enumerate(queries):
        logger.info(f"[JSMA] Query {i+1}/{len(queries)}")
        result = system.process_query(q["query"])

        metrics = compute_all_metrics(
            response=result["response"],
            ground_truth=q.get("ground_truth", {}),
            token_count=result["metrics"]["total_tokens"],
            message_trace=[],
        )
        results.append({
            "query_id": q.get("id"),
            "category": q.get("category"),
            "query": q["query"],
            "response": result["response"],
            "metrics": metrics,
            "raw_metrics": result["metrics"],
        })

    return results


def run_single_agent(config: dict, queries: list[dict]) -> list[dict]:
    """Run Single-Agent baseline on all queries."""
    neo4j_cfg = config.get("neo4j", {})
    db = GraphDB(
        uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
        user=neo4j_cfg.get("user", "neo4j"),
        password=neo4j_cfg.get("password", ""),
    )
    llm_cfg = config.get("llm", {})
    llm = LLMClient(api_key=llm_cfg.get("api_key", ""), model=llm_cfg.get("model", "gpt-4"))

    system = SingleAgentSystem(db=db, llm=llm)
    results = []

    for i, q in enumerate(queries):
        logger.info(f"[Single] Query {i+1}/{len(queries)}")
        result = system.process_query(q["query"])

        metrics = compute_all_metrics(
            response=result["response"],
            ground_truth=q.get("ground_truth", {}),
            token_count=result["metrics"]["total_tokens"],
            message_trace=[],
        )
        results.append({
            "query_id": q.get("id"),
            "category": q.get("category"),
            "query": q["query"],
            "response": result["response"],
            "metrics": metrics,
            "raw_metrics": result["metrics"],
        })

    return results


SYSTEM_RUNNERS = {
    "g2cp": run_g2cp,
    "ftma": run_ftma,
    "jsma": run_jsma,
    "single": run_single_agent,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="G²CP Evaluation Runner")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--system", choices=list(SYSTEM_RUNNERS.keys()) + ["all"], default="all")
    parser.add_argument("--queries", default="data/queries/test_queries.json")
    parser.add_argument("--output", default="results")
    parser.add_argument("--n", type=int, default=None, help="Number of queries to evaluate")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load queries
    with open(args.queries) as f:
        queries = json.load(f)
    if args.n:
        queries = queries[: args.n]
    logger.info(f"Loaded {len(queries)} queries")

    # Determine which systems to run
    systems = list(SYSTEM_RUNNERS.keys()) if args.system == "all" else [args.system]

    all_summaries: dict[str, dict] = {}

    for system_name in systems:
        logger.info(f"\n{'='*60}\nRunning {system_name.upper()}\n{'='*60}")
        start = time.time()

        runner = SYSTEM_RUNNERS[system_name]
        results = runner(config, queries)

        elapsed = time.time() - start

        # Save detailed results
        results_file = output_dir / f"{system_name}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Compute aggregated metrics
        all_metrics = [r["metrics"] for r in results]
        summary = aggregate_metrics(all_metrics)
        summary["total_time_s"] = round(elapsed, 2)
        summary["n_queries"] = len(results)
        all_summaries[system_name] = summary

        logger.info(f"\n{system_name.upper()} Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

    # Save comparison summary
    summary_file = output_dir / "comparison_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    logger.info(f"\nComparison saved to {summary_file}")

    # Print comparison table
    if len(all_summaries) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80)
        headers = ["Metric"] + [s.upper() for s in all_summaries.keys()]
        print(f"{'Metric':<30}", end="")
        for s in all_summaries:
            print(f"{s.upper():>12}", end="")
        print()
        print("-" * (30 + 12 * len(all_summaries)))

        for metric in ["accuracy_mean", "tokens_mean", "hallucination_rate_mean",
                        "cascading_error_rate_mean", "auditability_mean"]:
            print(f"{metric:<30}", end="")
            for s in all_summaries:
                val = all_summaries[s].get(metric, 0)
                print(f"{val:>12.4f}", end="")
            print()


if __name__ == "__main__":
    main()
