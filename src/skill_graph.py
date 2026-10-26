"""Skill graph and career requirements for the Data Scientist goal.
The edges are illustrative and should be reviewed by a subject expert."""
import networkx as nx

# Required mastery level (0-100) for each skill
REQUIRED = {
    "Python": 80,
    "SQL": 75,
    "Statistics": 75,
    "Probability": 70,
    "ML": 70,
    "DataViz": 70,
}

# How important each skill is for the Data Scientist goal (0 to 1)
GOAL_RELEVANCE = {
    "Python": 1.0,
    "SQL": 0.8,
    "Statistics": 1.0,
    "Probability": 0.9,
    "ML": 1.0,
    "DataViz": 0.6,
}

# Edge A -> B means "A is a prerequisite of B"
EDGES = [
    ("Python", "ML"),
    ("Python", "DataViz"),
    ("SQL", "DataViz"),
    ("Statistics", "Probability"),
    ("Statistics", "ML"),
    ("Probability", "ML"),
]


def build_graph():
    G = nx.DiGraph()
    G.add_nodes_from(REQUIRED.keys())
    G.add_edges_from(EDGES)
    return G