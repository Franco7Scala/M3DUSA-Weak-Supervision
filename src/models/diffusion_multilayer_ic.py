import random
from network_diffusion.models import BaseModel, NetworkUpdateBuffer

from src.support.utils import get_time_in_millis

"""
Multilayer Independent Cascade model

Features:
      - Each layer has its own activation probability (layer_probs[layer]).
      - Edge weights scale activation probability (p_eff = p_layer * weight).
      - OR logic across layers (infection possible from any layer).
      - No recovery (SI-like independent cascade).
      - Seeds are provided manually (seed_nodes).
      
Optimized model that:
    - Accepts precomputed neighbor_probs: dict[layer][u] = [(v, p_eff), ...]
    - Maintains internal multilayer states in self._local_states
    - Implements reset(seed_nodes=...) so it can be reused many times
"""


class MultiLayerICModel(BaseModel):

    def __init__(self, layer_probs, neighbor_probs, nodes_in_layer, seed_nodes=None):
        super().__init__()
        self.layer_probs = dict(layer_probs)
        self._provided_seeds = set(seed_nodes) if seed_nodes else set()

        # precomputed adjacency with effective probabilities:
        # neighbor_probs[layer_id][u] = list of (v, p_eff)
        self.neighbor_probs = neighbor_probs

        # nodes_in_layer[layer_id] = set(nodes present in that layer)
        self.nodes_in_layer = nodes_in_layer

        self._local_states = {}  # {node: {layer: "S"/"I"}}
        self.active = set()


    @property
    def _compartmental_graph(self):
        return None

    @property
    def _seed_selector(self):
        return None

    """
    Called once before simulation. Must return a list of NetworkUpdateBuffer.

    We set:
      - every node as S in layers where it exists
      - seeds as I in all layers they appear in
    """
    def determine_initial_states(self, network):

        start_time = get_time_in_millis()

        self._local_states = {}
        self.active = set()
        updates = []

        all_nodes = set()
        for layer_nodes in self.nodes_in_layer.values():
            all_nodes |= set(layer_nodes)

        # Initialize all nodes as S
        for node in all_nodes:
            layer_states = {}
            for layer_id, nodes in self.nodes_in_layer.items():
                if node in nodes:
                    layer_states[layer_id] = "S"
            self._local_states[node] = layer_states

        # Apply seeds as I
        for seed in self._provided_seeds:
            if seed in self._local_states:
                for layer_name in self._local_states[seed]:
                    self._local_states[seed][layer_name] = "I"
                self.active.add(seed)

        # Build initial update buffers
        for node, states in self._local_states.items():
            for layer_name, layer_state in states.items():
                updates.append(NetworkUpdateBuffer(node, layer_name, layer_state))

        print(f"determine_initial_states: {get_time_in_millis()-start_time}")

        return updates


    """
    Decide next state for (actor, layer_name).
    Returns "S" or "I", or None if actor not in this layer.

    IC rule:
      - If already I -> stays I.
      - If S -> check neighbors in any layer:
            success probability p_eff = layer_prob * edge_weight
        If ANY layer produces a successful attempt, actor becomes I in ALL layer.
    """
    def agent_evaluation_step(self, actor, layer_name, network):

        start_time = get_time_in_millis()

        # If actor not in this layer, skip
        if actor not in network.layers[layer_name]:
            return None

        current_state = self._local_states[actor].get(layer_name, "S")
        if current_state == "I":
            return "I"

        # Check neighbors across layers (OR logic)
        for l_name, nbr_map in self.neighbor_probs.items():
            if actor not in self.nodes_in_layer.get(l_name, ()):
                continue

            # neighbors that can infect actor are those with actor as target in nbr_map
            # but we stored neighbors per source: nbr_map[src] = [(dst, p)]
            # so we iterate sources that have edges to actor — we must check all sources
            # For speed we simply iterate over sources in nbr_map and check their targets.
            # This is still faster than repeated graph lookups because p_eff precomputed.
            for src, targets in nbr_map.items():
                # src must be infected in its layer to try
                if src not in self._local_states:
                    continue
                if self._local_states[src].get(l_name) != "I":
                    continue

                # check if src actually connects to actor in this layer (targets list)
                # targets is a list of (dst, p_eff)
                # we can iterate targets and test dst == actor
                for dst, p_eff in targets:
                    if dst != actor:
                        continue
                    # perform trial
                    if random.random() < p_eff:
                        return "I"

        # no infection
        return "S"

    """
    Evaluate the entire network for one epoch.
    If a node becomes infected in any layer, mark it "I" in all layers.
    """
    def network_evaluation_step(self, network):

        start_time = get_time_in_millis()

        updates = []
        newly_infected_global = set()

        all_nodes = set()
        for nodes in self.nodes_in_layer.values():
            all_nodes |= set(nodes)

        # First pass: detect all newly infected nodes
        for actor in all_nodes:

            # skip if already globally infected
            if actor in self.active:
                continue
            current = self._local_states[actor]
            # OR logic across layers
            for layer_name in current.keys():
                new_state = self.agent_evaluation_step(actor, layer_name, network)
                if new_state == "I" and current[layer_name] == "S":
                    newly_infected_global.add(actor)
                    break  # infected in any layer --> globally infected

        # Apply infection globally and generate per-layer update buffers
        for actor in newly_infected_global:

            for layer_name in self._local_states[actor].keys():
                self._local_states[actor][layer_name] = "I"
                updates.append(NetworkUpdateBuffer(actor, layer_name, "I"))

            # Track globally active nodes
            self.active.add(actor)

        return updates


    def get_allowed_states(self, network=None):
        #return {layer: ["S", "I"] for layer in self.layer_probs.keys()}
        #return ["S", "I"]
        return {
            layer: {
                "S": {"color": "blue"},
                "I": {"color": "red"}
            }
            for layer in self.layer_probs.keys()
        }

    def __str__(self):
        return f"MultiLayerICModel(layer_probs={self.layer_probs})"

    def reset(self, seed_nodes=None):
        self._provided_seeds = set(seed_nodes) if seed_nodes else set()
        self._local_states = {}
        self.active = set()