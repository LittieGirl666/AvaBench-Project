from __future__ import annotations

from ..suite import build_suite


def build_resources(*, seed: int = 0):
    suite = build_suite(domains=["workforce_access"], seed=seed)
    return suite.resources, suite.resource_rows

