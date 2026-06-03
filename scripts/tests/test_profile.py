from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from nctl_bench._profile import (  # noqa: E402
    ProfileTimer,
    configure_profiler,
    format_profile_table,
    get_profiler,
    prof,
    reset_profiler,
    write_profile_json,
)


@pytest.fixture(autouse=True)
def _profiler_reset_and_disable():
    """Tests must NOT leak profiler state between cases.  Disable + reset
    before AND after every test so any test that flips the singleton on
    can't poison the next one."""
    p = get_profiler()
    p.configure(enabled=False, sync_cuda=False)
    p.reset()
    yield
    p.configure(enabled=False, sync_cuda=False)
    p.reset()


def test_prof_is_no_op_when_disabled() -> None:
    p = get_profiler()
    assert p.enabled is False

    # The default fast path must be a yield-only context manager so it
    # cannot accumulate counters.  Verify that entering it many times
    # changes nothing.
    for _ in range(100):
        with prof("never.recorded"):
            pass

    snap = p.snapshot()
    assert snap == {}


def test_prof_accumulates_when_enabled_with_sync_off() -> None:
    configure_profiler(enabled=True, sync_cuda=False)
    p = get_profiler()

    with prof("section.a", sync=False):
        time.sleep(0.005)
    with prof("section.a", sync=False):
        time.sleep(0.005)
    with prof("section.b", sync=False):
        time.sleep(0.002)

    snap = p.snapshot()
    assert set(snap.keys()) == {"section.a", "section.b"}
    assert snap["section.a"]["calls"] == 2
    assert snap["section.b"]["calls"] == 1
    # Loose lower bound -- we slept 5ms twice, so total_s >= ~10ms minus
    # scheduler noise.  Use a generous floor that won't be flaky.
    assert snap["section.a"]["total_s"] >= 0.005
    assert snap["section.b"]["total_s"] >= 0.001
    # mean = total / calls
    assert snap["section.a"]["mean_s"] == pytest.approx(
        snap["section.a"]["total_s"] / 2.0
    )
    # max >= mean
    assert snap["section.a"]["max_s"] >= snap["section.a"]["mean_s"]


def test_profile_timer_min_tracks_smallest_observation() -> None:
    t = ProfileTimer()
    # Direct API: bypass the global singleton so this test is hermetic.
    t.add("s", 0.10)
    t.add("s", 0.05)
    t.add("s", 0.20)
    snap = t.snapshot()
    assert snap["s"]["min_s"] == pytest.approx(0.05)
    assert snap["s"]["max_s"] == pytest.approx(0.20)
    assert snap["s"]["calls"] == 3
    assert snap["s"]["mean_s"] == pytest.approx(0.35 / 3.0)


def test_reset_clears_all_sections() -> None:
    configure_profiler(enabled=True)
    with prof("section.x", sync=False):
        time.sleep(0.001)
    assert get_profiler().snapshot() != {}
    reset_profiler()
    assert get_profiler().snapshot() == {}


def test_write_profile_json_round_trip(tmp_path: Path) -> None:
    configure_profiler(enabled=True, sync_cuda=False)
    with prof("alpha", sync=False):
        time.sleep(0.001)
    with prof("beta", sync=False):
        time.sleep(0.001)

    out = tmp_path / "subdir" / "profile.json"  # tests that parents are created
    write_profile_json(out)

    payload = json.loads(out.read_text())
    assert payload["enabled"] is True
    assert payload["sync_cuda"] is False
    assert set(payload["sections"].keys()) == {"alpha", "beta"}


def test_format_profile_table_sorts_descending_and_caps_top() -> None:
    t = ProfileTimer()
    t.add("c.small", 0.001)
    t.add("a.big", 1.0)
    t.add("b.medium", 0.1)
    # Use the timer's own format_table so we test the implementation
    # directly without poisoning the global singleton.
    table = t.format_table()
    lines = table.splitlines()
    # Header + separator + 3 data rows.
    assert len(lines) == 5
    # First data row is the largest section.
    assert lines[2].startswith("a.big ")
    assert lines[3].startswith("b.medium ")
    assert lines[4].startswith("c.small ")
    # top= caps it.
    table_top1 = t.format_table(top=1)
    assert len(table_top1.splitlines()) == 3  # header + sep + 1 row
    assert table_top1.splitlines()[2].startswith("a.big ")


def test_format_profile_table_handles_empty_state() -> None:
    table = format_profile_table()
    assert "no sections" in table


def test_prof_records_even_when_exception_raised() -> None:
    """A section's elapsed time must be accumulated even if the bracketed
    code raises -- otherwise an exception-heavy code path would show as
    zero in the profile and silently hide the hot spot.
    """
    configure_profiler(enabled=True, sync_cuda=False)

    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom):
        with prof("oops", sync=False):
            time.sleep(0.002)
            raise Boom("intended")

    snap = get_profiler().snapshot()
    assert snap["oops"]["calls"] == 1
    assert snap["oops"]["total_s"] >= 0.001
