"""Checkpoint store pruning rides the gateway housekeeping tick, not the constructor.

Its ``git gc`` repacks the whole store (tens of seconds on a GB store); run at construction it
delayed the control socket, adapters and the code_sha stamp, so the first restart of the day (the
``hermes update`` one) looked hung and failed fleet verification.
"""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import gateway.run as gateway_run


class _OneTickStopEvent:
    def __init__(self):
        self.waited = False

    def is_set(self):
        return self.waited

    def wait(self, timeout=None):
        self.waited = True
        return True


class _Ticks:
    """Stop event that lets the housekeeping loop run exactly ``n`` ticks with no sleeping."""

    def __init__(self, n: int):
        self.n, self.left = 0, n

    def is_set(self):
        return self.n >= self.left

    def wait(self, timeout=None):
        self.n += 1
        return True


def test_gateway_housekeeping_runs_the_checkpoint_prune(monkeypatch):
    import tools.checkpoint_maintenance as cm

    calls = []
    monkeypatch.setattr(cm, "auto_prune_from_config", lambda: calls.append(True) or {"skipped": False})

    gateway_run._start_gateway_housekeeping(_OneTickStopEvent(), interval=0)

    assert calls == [True]


def _named_profile(home: Path, base_url: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("model:\n  provider: nous\n", encoding="utf-8")
    (home / ".env").write_text(f"NOUS_INFERENCE_BASE_URL={base_url}\n", encoding="utf-8")
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}), encoding="utf-8")


def test_multiplexed_checkpoint_prune_tick_sweeps_every_served_profile_store(tmp_path, monkeypatch):
    """The checkpoint prune tick reaches each served profile's OWN store.

    The store is per profile home; the tick ran unscoped, so a multiplexed gateway pruned only
    the launch home's store while every served profile's store grew without bound — a profile
    hit 1 GB across dozens of projects while the launch home's ``.last_prune`` stayed fresh.
    Real stores, real config files, real retention sweep: nothing here is patched.
    """
    from agent.secret_scope import set_multiplex_active
    from hermes_constants import get_hermes_home

    fake_home = tmp_path / "home"
    a = fake_home / ".hermes"
    b = a / "profiles" / "b"
    _named_profile(a, "https://a.example/v1")
    _named_profile(b, "https://b.example/v1")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HERMES_HOME", str(a))

    stale = time.time() - 2 * 86400
    for home in (a, b):
        (home / "config.yaml").write_text(
            "model:\n  provider: nous\n"
            "checkpoints:\n"
            "  enabled: true\n"
            "  auto_prune: true\n"
            "  retention_days: 1\n"
            "  min_interval_hours: 0\n",
            encoding="utf-8")
        base = home / "checkpoints"
        (base / "legacy-old").mkdir(parents=True)
        os.utime(base / "legacy-old", (stale, stale))  # past the 1-day retention cutoff
        (base / "legacy-fresh").mkdir(parents=True)    # today: retention keeps it

    set_multiplex_active(True)
    try:
        gateway_run._start_gateway_housekeeping(_Ticks(60), interval=0, runner=SimpleNamespace(
            config=SimpleNamespace(multiplex_profiles=True)))
    finally:
        set_multiplex_active(False)

    for home in (a, b):
        base = home / "checkpoints"
        assert not (base / "legacy-old").exists(), f"stale archive survived in {home}"
        assert (base / "legacy-fresh").exists(), f"fresh archive deleted in {home}"
        assert (base / ".last_prune").exists(), f"no prune marker written in {home}"
    assert get_hermes_home() == a  # the ambient home is untouched afterwards
