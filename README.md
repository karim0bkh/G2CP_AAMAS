# G²CP: Graph-Grounded Communication Protocol

[![AAMAS 2026](https://img.shields.io/badge/AAMAS-2026-blue)](https://aamas2026.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **A structured agent communication language where messages are graph operations rather than free text.**

G²CP replaces natural language inter-agent communication with explicit traversal commands, subgraph fragments, and update operations over a shared knowledge graph — enabling verifiable reasoning traces and eliminating ambiguity.

## Key Results

| Metric | FTMA | JSMA | Single | **G²CP** |
|--------|------|------|--------|----------|
| Task Accuracy | 0.67 | 0.74 | 0.71 | **0.90** |
| Tokens/Query | 2,847 | 2,134 | 1,456 | **768** |
| Hallucination Rate | 0.23 | 0.18 | 0.14 | **0.02** |
| Cascading Errors | 0.31 | 0.19 | 0.00 | **0.00** |
| Auditability | 0.42 | 0.68 | 1.00 | **1.00** |

## Architecture

```
User Query (NL) ──► Dispatcher ──► G²CP Messages ──► Specialist Agents
                                                           │
                        ┌──────────────────────────────────┘
                        │
                  ┌─────┴─────┐
                  │ Neo4j KG  │  ◄── Shared Knowledge Graph
                  └─────┬─────┘
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
          Diagnostic  Procedural  Synthesis
          Agent       Agent       Agent
```

## Quick Start

### Prerequisites

- Python 3.10+
- Neo4j 5.x (Community or Enterprise)
- OpenAI API key (for GPT-4) or compatible endpoint

### Installation

```bash
git clone https://github.com/<<anonymous>>/g2cp.git
cd g2cp
pip install -e ".[dev]"
```

### Configuration

```bash
cp configs/config.example.yaml configs/config.yaml
# Edit configs/config.yaml with your Neo4j and LLM credentials
```

### Load Knowledge Graph

```bash
# Start Neo4j, then:
python scripts/load_knowledge_graph.py --config configs/config.yaml
```

### Run G²CP System

```bash
# Interactive mode
python -m g2cp.main --config configs/config.yaml --mode interactive

# Batch evaluation
python -m g2cp.main --config configs/config.yaml --mode batch --queries data/queries/test_queries.json
```

### Run Evaluation

```bash
# Full evaluation across all systems
python evaluation/run_evaluation.py --config configs/config.yaml

# Specific system
python evaluation/run_evaluation.py --config configs/config.yaml --system g2cp

# Generate paper figures
python evaluation/generate_figures.py --results results/
```

## Repository Structure

```
g2cp/
├── g2cp/                       # Core library
│   ├── protocol/               # G²CP protocol definition
│   │   ├── messages.py         # Message types, performatives
│   │   ├── operations.py       # Graph operations (TRAVERSE, UPDATE)
│   │   ├── parser.py           # G²CP message parser
│   │   ├── serializer.py       # Message serialization/deserialization
│   │   └── commitments.py      # Social commitment semantics
│   ├── agents/                 # Agent implementations
│   │   ├── base.py             # Base agent class
│   │   ├── dispatcher.py       # Dispatcher agent (query decomposition)
│   │   ├── diagnostic.py       # Diagnostic agent (symptom→fault)
│   │   ├── procedural.py       # Procedural agent (fault→action)
│   │   ├── synthesis.py        # Synthesis agent (pattern discovery)
│   │   └── ingestion.py        # Ingestion agent (graph updates)
│   ├── engine/                 # Runtime engine
│   │   ├── runtime.py          # G²CP runtime orchestrator
│   │   ├── executor.py         # Graph operation executor (Neo4j)
│   │   ├── resolver.py         # Node resolution pipeline
│   │   ├── security.py         # Auth, RBAC, trust propagation
│   │   └── audit.py            # Audit logging
│   ├── baselines/              # Baseline implementations
│   │   ├── ftma.py             # Free-Text Multi-Agent
│   │   ├── jsma.py             # JSON-Structured Multi-Agent
│   │   └── single_agent.py     # Single-Agent RAG
│   ├── utils/                  # Utilities
│   │   ├── llm.py              # LLM interface (OpenAI, Llama)
│   │   ├── graph_db.py         # Neo4j connection manager
│   │   ├── embeddings.py       # Sentence embeddings for entity linking
│   │   └── tokens.py           # Token counting (tiktoken)
│   └── main.py                 # Entry point
├── data/
│   ├── knowledge_graph/        # KG data files
│   │   ├── nodes.json          # Node definitions
│   │   ├── edges.json          # Edge definitions
│   │   └── schema.json         # Graph schema
│   └── queries/
│       ├── test_queries.json   # 500 test queries with ground truth
│       └── real_world.json     # 21 real-world cases
├── evaluation/
│   ├── run_evaluation.py       # Main evaluation script
│   ├── metrics.py              # Metric computation
│   └── generate_figures.py     # Paper figure generation
├── configs/
│   └── config.example.yaml     # Configuration template
├── tests/                      # Unit and integration tests
├── scripts/                    # Utility scripts
│   └── load_knowledge_graph.py # KG loader
├── pyproject.toml
├── LICENSE
└── README.md
```

## Citation

```bibtex
@inproceedings{benkhaled2026g2cp,
  title={G²CP: A Graph-Grounded Communication Protocol for Verifiable and Efficient Multi-Agent Reasoning},
  author={Ben Khaled, Karim and Monticolo, Davy and Mastagli, Maxime},
  booktitle={Proc. of the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2026)},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
