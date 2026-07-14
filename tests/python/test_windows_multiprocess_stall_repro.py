"""Reproducer for the multi-hour Windows CI stall observed in Speasy's
``CacheRequestsDeduplicationMultiProcess`` test (SciQLop/speasy PR #315).

That test spawns 4 brand-new ``multiprocessing.Process`` per step (100
steps), each opening a fresh ``Cache`` connection to the same shared file
and doing one cheap operation. On Windows this made the whole test take
2+ hours; on Linux (fork or spawn start method) it takes seconds.

Bisection in the Speasy repo (2026-07-14) ruled out: process-spawn cost
alone (~0.09s/step for a no-op spawn), generic heavy-native-import cost
(numpy+pandas alone: ~0.7s/step), and the test's own dedup/cleanup logic
(reproduces with zero speasy-specific code). It isolated the cost to
opening a ``pysciqlop_cache`` connection from a freshly spawned process
while 3 others do the same concurrently against the same file — even a
bare ``Cache(path)`` + ``.get()`` with no write reproduced ~10-20s per
open on Windows. Locally reproducing with ``spawn`` on Linux stayed fast
(~0.06s/step for 4 processes), which rules out fork-vs-spawn as the
mechanism and points at Windows-specific SQLite/WAL/mmap behavior
(``PRAGMA mmap_size=268435456`` + ``busy_timeout=600000`` are the leading
suspects — see CLAUDE.md "WAL mode + 600s busy_timeout for multi-process
safety").

This test intentionally uses a fresh ``Process()`` per step (matching the
real Speasy test), not a persistent ``Pool``, and calls ``incr`` (a real
write) to match ``MPDataProvider.increase_count``. It is not meant to run
in the always-on suite (100 steps would be far too slow if it ever
regresses) — see the reduced-step, always-on variant below for CI
coverage plus a max-step-time assertion to catch a regression by contract
rather than by silent multi-hour hang.
"""
import shutil
import tempfile
import time
import unittest
from multiprocessing import Process

from pysciqlop_cache import Cache


def _open_and_incr(path, key):
    cache = Cache(path)
    cache.incr(key, 1, default=0)


class TestWindowsMultiprocessStallRepro(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_fresh_process_per_step_stays_fast(self):
        """Regression guard: N steps of 4 fresh-process opens must stay
        fast. Fails loudly (assertion) instead of silently stalling for
        hours if the Windows connection-open contention regresses."""
        n_steps = 10
        max_seconds_per_step = 2.0

        t0 = time.monotonic()
        for step in range(n_steps):
            key = f"counter::{step}"
            processes = [
                Process(target=_open_and_incr, args=(self.tmp_dir, key))
                for _ in range(4)
            ]
            for p in processes:
                p.start()
            for p in processes:
                p.join()

        elapsed = time.monotonic() - t0
        per_step = elapsed / n_steps
        self.assertLess(
            per_step, max_seconds_per_step,
            f"{n_steps} steps x 4 fresh-process cache opens took "
            f"{elapsed:.1f}s ({per_step:.2f}s/step) - expected under "
            f"{max_seconds_per_step}s/step. This is the exact pattern "
            f"behind the Speasy multiprocess Windows CI stall.",
        )


if __name__ == "__main__":
    unittest.main()
