#!/usr/bin/env python3
"""Build the yearly directed, weighted artist collaboration networks.

Built from data/dataset.csv: nodes are artists, and an edge A -> B means
original artist A (main_artist_group) featured guest artist B
(collaboration_groups) in a challenge Short; its weight is the number of such
ties.  Networks for 2022, 2023 and 2024 (year of published_at) share node IDs,
so motif counts are comparable across years.

Output: network_modeling/networks/
  artist_mapping.txt                node ID -> artist (IDs = alphabetical order)
  <year>/edges_<year>.txt           "from to" per directed edge (motif analysis input)
  <year>/edges_<year>_weighted.txt  From, To, Weight
Usage : python network_modeling/01_network_construction.py
"""
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "network_modeling" / "networks"
YEARS = ["2022", "2023", "2024"]


def load_ties():
    d = pd.read_csv(REPO / "data" / "dataset.csv")
    d["year"] = d.published_at.str[:4]
    return d[["main_artist_group", "collaboration_groups", "year"]]


def build_graph(ties):
    """Directed graph with weight = number of Shorts featuring the guest."""
    g = nx.DiGraph()
    for (a, b), w in Counter(zip(ties.main_artist_group, ties.collaboration_groups)).items():
        g.add_edge(a, b, weight=w)
    return g


def main():
    ties = load_ties()
    artists = sorted(set(ties.main_artist_group) | set(ties.collaboration_groups))
    node_id = {a: i for i, a in enumerate(artists)}
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"NodeID": range(len(artists)), "ArtistName": artists}).to_csv(
        OUT / "artist_mapping.txt", sep="\t", index=False)

    print(f"{'year':<6} {'nodes':>6} {'edges':>6} {'ties':>6} {'reciprocal pairs':>17}")
    for year in YEARS:
        g = build_graph(ties[ties.year.eq(year)])
        folder = OUT / year
        folder.mkdir(exist_ok=True)
        edges = [(node_id[a], node_id[b], w) for a, b, w in g.edges(data="weight")]
        (folder / f"edges_{year}.txt").write_text("".join(f"{a} {b}\n" for a, b, _ in edges))
        pd.DataFrame(edges, columns=["From", "To", "Weight"]).to_csv(
            folder / f"edges_{year}_weighted.txt", sep="\t", index=False)
        reciprocal = sum(g.has_edge(b, a) for a, b in g.edges()) // 2
        print(f"{year:<6} {g.number_of_nodes():>6} {g.number_of_edges():>6} "
              f"{int(g.size(weight='weight')):>6} {reciprocal:>17}")


if __name__ == "__main__":
    main()
