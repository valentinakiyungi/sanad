"""Export the knowledge graph as Cypher, to explore it in Neo4j Browser.

python scripts/export_neo4j.py .index/graph.json > graph.cypher
cat graph.cypher | cypher-shell -u neo4j -p <password>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def q(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def main(path: str) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    label = {"law": "Law", "article": "Article", "term": "Term"}
    print("MATCH (n) WHERE n:Law OR n:Article OR n:Term DETACH DELETE n;")
    for node in data["nodes"]:
        kind = node.get("kind")
        if kind not in label:
            continue
        props = {"id": node["id"]}
        if kind == "article":
            props |= {"law": node["law"], "number": node["number"], "title": node.get("title", "")}
        elif kind == "term":
            props["label"] = node["label"]
        elif kind == "law":
            props |= {f"title_{k}": v for k, v in node.get("titles", {}).items()}
        body = ", ".join(f"{k}: {q(v)}" for k, v in props.items())
        print(f"CREATE (:{label[kind]} {{{body}}});")
    for edge in data["edges"]:
        print(f"MATCH (a {{id: {q(edge['source'])}}}), (b {{id: {q(edge['target'])}}}) CREATE (a)-[:{edge['key']}]->(b);")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".index/graph.json")
