from __future__ import annotations

from ..suite import build_suite


def build_operations(*, seed: int = 0):
    suite = build_suite(domains=["compliance_audit"], seed=seed)
    return suite.bases, suite.subs

