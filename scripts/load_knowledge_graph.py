"""Load knowledge graph data into Neo4j.

Usage:
    python scripts/load_knowledge_graph.py --config configs/config.yaml
    python scripts/load_knowledge_graph.py --config configs/config.yaml --clear
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from g2cp.utils.graph_db import GraphDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("kg_loader")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load knowledge graph into Neo4j")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--clear", action="store_true", help="Clear existing data first")
    parser.add_argument("--data-dir", default="data/knowledge_graph")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    neo4j_cfg = config.get("neo4j", {})
    db = GraphDB(
        uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
        user=neo4j_cfg.get("user", "neo4j"),
        password=neo4j_cfg.get("password", ""),
        database=neo4j_cfg.get("database", "neo4j"),
    )

    data_dir = Path(args.data_dir)

    # Clear if requested
    if args.clear:
        logger.info("Clearing existing graph data...")
        db.clear()

    # Load nodes
    nodes_file = data_dir / "nodes.json"
    if nodes_file.exists():
        with open(nodes_file) as f:
            nodes = json.load(f)
        logger.info(f"Loading {len(nodes)} nodes...")
        count = db.load_nodes(nodes)
        logger.info(f"Loaded {count} nodes")
    else:
        logger.warning(f"Nodes file not found: {nodes_file}")

    # Load edges
    edges_file = data_dir / "edges.json"
    if edges_file.exists():
        with open(edges_file) as f:
            edges = json.load(f)
        logger.info(f"Loading {len(edges)} edges...")
        count = db.load_edges(edges)
        logger.info(f"Loaded {count} edges")
    else:
        logger.warning(f"Edges file not found: {edges_file}")

    # Print schema
    schema = db.get_schema()
    logger.info(f"\nGraph Schema:")
    logger.info(f"  Node types: {schema.get('node_types', [])}")
    logger.info(f"  Edge types: {schema.get('edge_types', [])}")
    logger.info(f"  Total nodes: {schema.get('node_count', 0)}")
    logger.info(f"  Total edges: {schema.get('edge_count', 0)}")

    db.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
