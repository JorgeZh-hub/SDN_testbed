# path_engine.py
from __future__ import annotations

import heapq
from typing import Dict, Tuple, List, Optional, Set, Callable

class PathEngine:
    """Dijkstra over the switch graph, with avoid_edges and residual constraint."""

    def __init__(self, topo, stats, logger):
        self.topo = topo
        self.stats = stats
        self.log = logger

    def shortest_path(self,
                      src: int,
                      dst: int,
                      avoid_edges: Optional[Set[Tuple[int,int]]] = None,
                      min_residual_mbps: float = 0.0,
                      cost_fn: Optional[Callable[[Tuple[int,int]], float]] = None) -> Optional[List[int]]:
        if src == dst:
            return [src]
        avoid = avoid_edges or set()
        if cost_fn is None:
            cost_fn = lambda e: 1.0

        dist: Dict[int, float] = {src: 0.0}
        prev: Dict[int, int] = {}
        pq = [(0.0, src)]

        while pq:
            d,u = heapq.heappop(pq)
            if u == dst:
                break
            if d != dist.get(u, float("inf")):
                continue
            for v, outp in self.topo.neigh.get(u, {}).items():
                e = (u,v)
                if e in avoid:
                    continue
                # residual constraint
                if min_residual_mbps > 0.0:
                    if self.stats.residual_mbps(e) < min_residual_mbps:
                        continue
                nd = d + float(cost_fn(e))
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if dst not in dist:
            return None
        # reconstruct
        path = [dst]
        cur = dst
        while cur != src:
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        return path
