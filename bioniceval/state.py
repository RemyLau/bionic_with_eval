from pathlib import Path
from typing import Dict, List, Union

import networkx as nx
import pandas as pd


class State:
    """Keeps track of the evaluation state."""

    config_path: Path
    config_name: Path
    config_features: List[Dict[str, Union[str, Path]]]
    config_networks: List[Dict[str, Union[str, Path]]]
    config_standards: List[Dict[str, Union[str, Path]]]
    result_path: Path = Path("bioniceval/results")
    consolidation: str
    plot: bool = True
    baseline: List[str] = []  # baseline genes list
    features: Dict[str, pd.DataFrame]  # actual feature sets evaluated by the library
    networks: Dict[str, nx.Graph]  # actual networks evaluated by the library
    consolidated_genes: List[
        str
    ]  # shared genes if `consolidation` == "intersection", all genes otherwise

    # results of the evaluation tasks
    coannotation_evaluations: pd.DataFrame
    module_detection_evaluations: pd.DataFrame
    function_prediction_evaluations: pd.DataFrame
