# path_engine.py
from dataclasses import dataclass
from collections import deque

@dataclass
class Path:
    path: list
    cost: float

class PathEngine:
    def __init__(self, logger, topo, bw_getter, default_bw=10000000, max_paths=1):
        """
        topo: TopologyManager
        bw_getter(u, out_port) -> bw_value  (tu bw[u][port])
        """
        self.log = logger
        self.topo = topo
        self.bw_getter = bw_getter
        self.default_bw = default_bw
        self.max_paths = max_paths

        self.path_table = {}            # key -> [Path] (optimal list)
        self.paths_table = {}           # key -> [Path] all
        self.path_with_ports_table = {} # key -> [ {sw: (in,out)} ]

    def _find_path_cost(self, nodes):
        cost = 0.0
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i+1]
            out_port = self.topo.neigh[u][v]
            cost += float(self.bw_getter(u, out_port) or self.default_bw)
        return cost

    def find_paths_and_costs(self, src, dst):
        if src == dst:
            return [Path([src], 0.0)]

        q = deque([(src, [src])])
        out = []

        while q:
            node, path = q.popleft()
            visited = set(path)

            for v in sorted(self.topo.neigh[node].keys()):
                if v in visited:
                    continue
                new_path = path + [v]
                if v == dst:
                    out.append(Path(new_path, self._find_path_cost(new_path)))
                else:
                    q.append((v, new_path))
        return out

    def find_n_optimal_paths(self, paths):
        return sorted(paths, key=lambda p: (p.cost, len(p.path), tuple(p.path)))[:self.max_paths]

    def add_ports_to_path(self, path_obj, first_port, last_port):
        nodes = path_obj.path
        bar = {}
        in_port = first_port
        for s1, s2 in zip(nodes[:-1], nodes[1:]):
            out_port = self.topo.neigh[s1][s2]
            bar[s1] = (in_port, out_port)
            in_port = self.topo.neigh[s2][s1]
        bar[nodes[-1]] = (in_port, last_port)
        return [bar]

    def ensure_path(self, src, first_port, dst, last_port):
        key = (src, first_port, dst, last_port)
        if key in self.path_table and key in self.path_with_ports_table:
            return key

        paths = self.find_paths_and_costs(src, dst)
        best = self.find_n_optimal_paths(paths)
        ports_map = self.add_ports_to_path(best[0], first_port, last_port)

        self.paths_table[key] = paths
        self.path_table[key] = best
        self.path_with_ports_table[key] = ports_map

        return key

    def invalidate_by_link(self, s1, s2):
        def uses(nodes):
            return any(
                (nodes[i] == s1 and nodes[i+1] == s2) or (nodes[i] == s2 and nodes[i+1] == s1)
                for i in range(len(nodes)-1)
            )

        affected = [k for k, best in self.path_table.items() if uses(best[0].path)]
        for k in affected:
            self.path_table.pop(k, None)
            self.paths_table.pop(k, None)
            self.path_with_ports_table.pop(k, None)
        return affected
