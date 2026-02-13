"""G²CP Main Entry Point.

Supports interactive mode and batch evaluation.

Usage:
    python -m g2cp.main --config configs/config.yaml --mode interactive
    python -m g2cp.main --config configs/config.yaml --mode batch --queries data/queries/test_queries.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from g2cp.agents.diagnostic import DiagnosticAgent
from g2cp.agents.dispatcher import DispatcherAgent
from g2cp.agents.ingestion import IngestionAgent
from g2cp.agents.procedural import ProceduralAgent
from g2cp.agents.synthesis import SynthesisAgent
from g2cp.engine.runtime import G2CPRuntime
from g2cp.engine.security import SecurityManager
from g2cp.utils.embeddings import EntityLinker
from g2cp.utils.graph_db import GraphDB
from g2cp.utils.llm import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("g2cp")


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config not found: {path}, using defaults")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_system(config: dict[str, Any]) -> G2CPRuntime:
    """Build the full G²CP system from configuration."""
    # Initialize components
    neo4j_cfg = config.get("neo4j", {})
    db = GraphDB(
        uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
        user=neo4j_cfg.get("user", "neo4j"),
        password=neo4j_cfg.get("password", ""),
        database=neo4j_cfg.get("database", "neo4j"),
    )

    llm_cfg = config.get("llm", {})
    llm = LLMClient(
        api_key=llm_cfg.get("api_key", ""),
        model=llm_cfg.get("model", "gpt-4"),
        base_url=llm_cfg.get("base_url"),
        temperature=llm_cfg.get("temperature", 0.0),
        max_tokens=llm_cfg.get("max_tokens", 1024),
    )

    emb_cfg = config.get("embeddings", {})
    linker = EntityLinker(
        threshold=emb_cfg.get("threshold", 0.85),
        model_name=emb_cfg.get("model", "all-MiniLM-L6-v2"),
    )

    # Load knowledge graph for entity linking index
    kg_dir = Path(config.get("data", {}).get("knowledge_graph_dir", "data/knowledge_graph"))
    nodes_file = kg_dir / "nodes.json"
    if nodes_file.exists():
        with open(nodes_file) as f:
            nodes = json.load(f)
        linker.build_index(nodes)
        db.load_nodes(nodes)
        logger.info(f"Loaded {len(nodes)} nodes into graph and entity index")

    edges_file = kg_dir / "edges.json"
    if edges_file.exists():
        with open(edges_file) as f:
            edges = json.load(f)
        db.load_edges(edges)
        logger.info(f"Loaded {len(edges)} edges into graph")

    # Build agents
    security = SecurityManager()
    agents = {
        "dispatcher": DispatcherAgent(db=db, llm=llm, linker=linker),
        "diagnostic": DiagnosticAgent(db=db, llm=llm),
        "procedural": ProceduralAgent(db=db, llm=llm),
        "synthesis": SynthesisAgent(db=db, llm=llm),
        "ingestion": IngestionAgent(db=db, llm=llm),
    }

    # Register agents with security
    for agent_id, agent in agents.items():
        security.register_agent(agent_id, agent.role)

    return G2CPRuntime(db=db, agents=agents, security=security)


def interactive_mode(runtime: G2CPRuntime) -> None:
    """Run G²CP in interactive chat mode."""
    print("\n╔══════════════════════════════════════════════╗")
    print("║   G²CP: Graph-Grounded Communication Protocol ║")
    print("║   Interactive Mode                             ║")
    print("╚══════════════════════════════════════════════╝\n")
    print("Type your query (or 'quit' to exit):\n")

    while True:
        try:
            query = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        result = runtime.process_query(query)
        print(f"\nG²CP > {result['response']}")
        print(f"\n  [Metrics: {result['metrics']['total_tokens']} tokens, "
              f"{result['metrics']['n_messages']} messages, "
              f"{result['metrics']['total_time_s']}s]\n")


def batch_mode(runtime: G2CPRuntime, queries_file: str, output_dir: str = "results") -> None:
    """Run G²CP in batch evaluation mode."""
    with open(queries_file) as f:
        queries = json.load(f)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    for i, q in enumerate(queries):
        query_text = q["query"]
        logger.info(f"Processing query {i+1}/{len(queries)}: {query_text[:60]}...")

        result = runtime.process_query(query_text)
        results.append({
            "query_id": q.get("id", f"Q-{i}"),
            "category": q.get("category", "unknown"),
            "query": query_text,
            "response": result["response"],
            "ground_truth": q.get("ground_truth", {}),
            "metrics": result["metrics"],
        })

    # Save results
    results_file = output_path / "g2cp_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    # Print summary
    total_tokens = sum(r["metrics"]["total_tokens"] for r in results)
    avg_tokens = total_tokens / max(len(results), 1)
    print(f"\n=== Batch Results ===")
    print(f"Queries processed: {len(results)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens/query: {avg_tokens:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="G²CP: Graph-Grounded Communication Protocol")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--mode", choices=["interactive", "batch"], default="interactive")
    parser.add_argument("--queries", default="data/queries/test_queries.json", help="Queries file for batch mode")
    parser.add_argument("--output", default="results", help="Output directory for results")
    args = parser.parse_args()

    config = load_config(args.config)
    runtime = build_system(config)

    if args.mode == "interactive":
        interactive_mode(runtime)
    elif args.mode == "batch":
        batch_mode(runtime, args.queries, args.output)


if __name__ == "__main__":
    main()
