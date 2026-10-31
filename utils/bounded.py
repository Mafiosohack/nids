"""A per-key state table with a ceiling: what every per-source tracker needs.

Why: a NIDS keeps a little state per source address, and the source address
is the one field an attacker sets for free. Spoofing a new source per packet
created a new tracker entry per packet, and pruning that only runs when the
same key returns never freed them (measured: ~1 KB/packet, unbounded).

Policy: least recently SEEN goes first. An attacker who keeps sending keeps
their entries hot; a flood of one-shot spoofed sources mostly evicts itself.
Eviction is O(1). Every eviction is counted, so the operator can see the sensor
is under state pressure (GET /status -> state_tables) rather than discovering
it from a missed detection.

Drop-in for the defaultdict usage in main.py: d[k] creates via the factory,
`k in d` does not create, plus get/pop/items/keys/values/len/clear.
"""

from collections import OrderedDict
from typing import Any, Callable, Hashable, Iterator, Optional

_MISSING = object()


class BoundedDict:
    __slots__ = ("_d", "factory", "max_keys", "name", "evictions")

    def __init__(self, factory: Optional[Callable[[], Any]] = None,
                 max_keys: int = 50_000, name: str = ""):
        if max_keys < 1:
            raise ValueError("max_keys must be >= 1")
        self._d: "OrderedDict[Hashable, Any]" = OrderedDict()
        self.factory = factory
        self.max_keys = int(max_keys)
        self.name = name
        self.evictions = 0

    # ── access ──────────────────────────────────────────────────────────────
    def __getitem__(self, key):
        d = self._d
        if key in d:
            d.move_to_end(key)
            return d[key]
        if self.factory is None:
            raise KeyError(key)
        value = self.factory()
        self._insert(key, value)
        return value

    def __setitem__(self, key, value) -> None:
        d = self._d
        if key in d:
            d.move_to_end(key)
            d[key] = value
        else:
            self._insert(key, value)

    def _insert(self, key, value) -> None:
        d = self._d
        while len(d) >= self.max_keys:
            d.popitem(last=False)
            self.evictions += 1
        d[key] = value

    def get(self, key, default=None):
        """Read without creating and without refreshing recency: a lookup is
        not evidence the source is still active."""
        return self._d.get(key, default)

    def pop(self, key, default=_MISSING):
        if default is _MISSING:
            return self._d.pop(key)
        return self._d.pop(key, default)

    def __delitem__(self, key) -> None:
        del self._d[key]

    def __contains__(self, key) -> bool:
        return key in self._d

    def __len__(self) -> int:
        return len(self._d)

    def __iter__(self) -> Iterator:
        return iter(list(self._d))          # snapshot: callers may delete while looping

    def keys(self):
        return list(self._d.keys())

    def values(self):
        return list(self._d.values())

    def items(self):
        return list(self._d.items())

    def clear(self) -> None:
        self._d.clear()

    # ── capacity ────────────────────────────────────────────────────────────
    def set_limit(self, max_keys: int) -> None:
        """Change the ceiling; shrinking evicts the least recently seen now."""
        if max_keys < 1:
            raise ValueError("max_keys must be >= 1")
        self.max_keys = int(max_keys)
        while len(self._d) > self.max_keys:
            self._d.popitem(last=False)
            self.evictions += 1

    def stats(self) -> dict:
        return {"size": len(self._d), "limit": self.max_keys, "evictions": self.evictions}

    def __repr__(self) -> str:
        return f"BoundedDict({self.name!r}, {len(self._d)}/{self.max_keys})"


def trim_oldest(table: dict, max_keys: int, age: Callable[[Any], float],
                headroom: float = 0.10, on_evict: Optional[Callable[[Any], None]] = None) -> int:
    """For detectors that key eviction on their own timestamps: when `table` is
    full, drop the oldest `headroom` fraction in ONE sort, not one entry per
    insert. Same oldest-first policy; the sort is paid once per ~10% of the
    table instead of on every new key (which made a full table a CPU sink).
    Returns how many were dropped."""
    if len(table) < max_keys:
        return 0
    n = max(1, int(max_keys * headroom)) + (len(table) - max_keys)
    victims = sorted(table, key=lambda k: age(table[k]))[:n]
    for k in victims:
        del table[k]
        if on_evict is not None:
            on_evict(k)
    return len(victims)
