"""Login throttling: slow down password guessing without locking anyone out.

Two counters, with deliberately different strengths:

* **Per source IP - strict.** One address guessing repeatedly is the ordinary
  brute force. After `ip_free` failures each further attempt must wait
  exponentially longer, up to `ip_max_delay` (5 minutes by default).

* **Per username - lenient.** Many addresses guessing one account (a spray, or
  a botnet) is only visible here. But this is also the counter an attacker can
  aim at the REAL admin to keep them out, so it is capped at `user_max_delay`
  (30 s): enough to make an online attack against one account hopeless, never
  enough to be a denial of service against the person who owns it.

There is no hard lockout anywhere. A lockout converts "attacker knows your
username" into "attacker can stop you logging in", which on a security tool is
the worse outcome.

Failures older than `window_sec` are forgotten. Memory is bounded by
`max_keys` per counter, evicting the least recently seen key, so rotating
source addresses cannot grow the tables without limit.
"""

import threading
import time
from collections import OrderedDict
from typing import Callable, List

DEFAULT_IP_FREE = 5          # typos allowed before any delay
DEFAULT_USER_FREE = 20
DEFAULT_IP_MAX_DELAY = 300.0
DEFAULT_USER_MAX_DELAY = 30.0
DEFAULT_WINDOW_SEC = 900.0
DEFAULT_MAX_KEYS = 10_000


class _Counter:
    """Failure timestamps per key, LRU-bounded."""

    def __init__(self, free: int, max_delay: float, window: float, max_keys: int):
        self.free, self.max_delay = int(free), float(max_delay)
        self.window, self.max_keys = float(window), int(max_keys)
        self._fails: "OrderedDict[str, List[float]]" = OrderedDict()

    def add(self, key: str, now: float) -> None:
        hist = [t for t in self._fails.pop(key, []) if now - t <= self.window]
        hist.append(now)
        self._fails[key] = hist[-64:]
        while len(self._fails) > self.max_keys:
            self._fails.popitem(last=False)

    def clear(self, key: str) -> None:
        self._fails.pop(key, None)

    def retry_after(self, key: str, now: float) -> float:
        hist = [t for t in self._fails.get(key, ()) if now - t <= self.window]
        excess = len(hist) - self.free
        if excess <= 0:
            return 0.0
        delay = min(self.max_delay, 2.0 ** (excess - 1))
        return max(0.0, hist[-1] + delay - now)

    def __len__(self) -> int:
        return len(self._fails)


class LoginThrottle:
    def __init__(self, clock: Callable[[], float] = time.time,
                 ip_free: int = DEFAULT_IP_FREE,
                 user_free: int = DEFAULT_USER_FREE,
                 ip_max_delay: float = DEFAULT_IP_MAX_DELAY,
                 user_max_delay: float = DEFAULT_USER_MAX_DELAY,
                 window_sec: float = DEFAULT_WINDOW_SEC,
                 max_keys: int = DEFAULT_MAX_KEYS):
        self.clock = clock
        self._lock = threading.Lock()
        self._ip = _Counter(ip_free, ip_max_delay, window_sec, max_keys)
        self._user = _Counter(user_free, user_max_delay, window_sec, max_keys)

    def retry_after(self, ip: str, username: str) -> float:
        """Seconds the caller must wait before this attempt is even checked."""
        now = self.clock()
        with self._lock:
            return max(self._ip.retry_after(ip or "?", now),
                       self._user.retry_after((username or "").lower(), now))

    def failure(self, ip: str, username: str) -> None:
        now = self.clock()
        with self._lock:
            self._ip.add(ip or "?", now)
            self._user.add((username or "").lower(), now)

    def success(self, ip: str, username: str) -> None:
        """A correct password clears that source and that account."""
        with self._lock:
            self._ip.clear(ip or "?")
            self._user.clear((username or "").lower())

    def reset(self) -> None:
        with self._lock:
            self._ip._fails.clear()
            self._user._fails.clear()

    @property
    def tracked_keys(self) -> int:
        return len(self._ip) + len(self._user)
