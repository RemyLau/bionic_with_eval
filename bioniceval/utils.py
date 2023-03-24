
import json
from functools import reduce
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional
from pathlib import Path

from state import State


def resolve_config_path(path: Path):
    if path == Path(path.name):
        path = Path("bioniceval/config") / path
    name = path.stem
    State.config_path = path
    State.config_name = name


def resolve_tasks() -> List[str]:
    tasks = list(set([standard["task"] for standard in State.config_standards]))
    return tasks


def import_datasets(consolidation: str = "union"):
    """Imports datasets and consolidates them to have the same number of genes,
    as given by the `consolidation` strategy.
    """

    features = {
        item["name"]: pd.read_csv(item["path"], delimiter=item["delimiter"], index_col=0)
        for item in State.config_features
    }

    networks = {
        item["name"]: nx.read_weighted_edgelist(item["path"], delimiter=item["delimiter"])
        for item in State.config_networks
    }
    consolidate_datasets(features, networks)


def import_standard():
    pass


def consolidate_datasets(
    features: Optional[Dict[str, pd.DataFrame]] = [], networks: Optional[Dict[str, nx.Graph]] = []
):
    consolidation = State.consolidation

    # compute consolidated genes
    feature_genes = [list(feat.index) for feat in features.values()]
    network_genes = [list(net.nodes()) for net in networks.values()]
    if consolidation == "union":
        consolidated_genes = reduce(np.union1d, feature_genes + network_genes)
    elif consolidation == "intersection":
        consolidated_genes = reduce(np.intersect1d, feature_genes + network_genes)
    elif (not State.baseline == []) and consolidation == "baseline":
        # if a baseline file is specified and the config consolidation mode is baseline
        consolidated_genes = State.baseline
    else:
        raise ValueError(f"Consolidation strategy '{consolidation}' is not supported.")

    features = consolidate_features(features, consolidated_genes)
    networks = consolidate_networks(networks, consolidated_genes)

    # test if features and networks are not empty
    if features and networks:
        assert list(list(features.values())[0].index) == list(list(networks.values())[0].nodes())

    State.features = features
    State.networks = networks
    State.consolidated_genes = consolidated_genes


def consolidate_features(
    features: Dict[str, pd.DataFrame], genes: List[str]
) -> Dict[str, pd.DataFrame]:
    """Consolidates features by ensuring they share the same genes in the same order."""

    consolidated = {}
    for name, feat in features.items():
        consolidated[name] = feat.reindex(genes).fillna(0)
    return consolidated


def consolidate_networks(networks: Dict[str, nx.Graph], genes: List[str]) -> Dict[str, nx.Graph]:
    """Consolidates networks by ensuring they share the same genes in the same order."""

    consolidated = {}
    for name, net in networks.items():
        new_net = nx.Graph()
        new_net.add_nodes_from(genes)
        new_net.add_edges_from(net.subgraph(genes).edges(data=True))
        consolidated[name] = new_net
    return consolidated


def process_config(exclude_tasks: List[str], exclude_standards: List[str], baseline_path: Path):
    with State.config_path.open() as f:
        config = json.load(f)
        if not baseline_path == Path(''):
            # Read each row of the baseline genes list and save them to a list if a baseline file is specified
            State.baseline = [line.strip() for line in baseline_path.open()]
        for key, value in config.items():
            if key == "standards":
                # filter out excluded tasks and standards
                value = [
                    standard
                    for standard in value
                    if standard["task"] not in exclude_tasks
                    and standard["name"] not in exclude_standards
                ]

            if key == "features":
                State.config_features = value
            if key == "networks":
                State.config_networks = value
            if key == "standards":
                State.config_standards = value
            if key == "consolidation":
                State.consolidation = value
            if key == "plot":
                State.plot = value
            if key == "result_path":
                State.result_path = value
