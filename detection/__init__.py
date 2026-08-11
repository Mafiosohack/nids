"""Testable detection primitives, deliberately free of scapy/FastAPI imports.

`main.py` is the wiring layer: it owns packet capture and the HTTP API. Everything
in this package is pure logic with an injectable clock, so the efficacy harness
and the unit tests can drive it deterministically without a network.

The one exception is `reassembly.scapy_defragment`, which is an optional thin
adapter guarded by a try/except import.
"""
