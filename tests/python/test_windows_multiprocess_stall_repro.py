"""Reproducer for the multi-hour Windows CI stall observed in Speasy's
``CacheRequestsDeduplicationMultiProcess`` test (SciQLop/speasy PR #315).

That test spawns 4 brand-new ``multiprocessing.Process`` per step (100
steps), each opening a fresh ``Cache`` connection to the same shared file
and doing one cheap operation. On Windows this made the whole test take
2+ hours; on Linux (fork or spawn start method) it takes seconds.

Bisection in the Speasy repo (2026-07-14) ruled out: process-spawn cost
alone (~0.09s/step for a no-op spawn), generic heavy-native-import cost
(numpy+pandas alone: ~0.7s/step), and the test's own dedup/cleanup logic
(reproduces with zero speasy-specific code). It pointed at opening a
``pysciqlop_cache`` connection from a freshly spawned process while 3
others do the same concurrently against the same file — even a bare
``from speasy.core.cache import _cache`` import (no explicit read/write)
reproduced ~10-20s per open on Windows.

**But** a bare ``pysciqlop_cache.Cache(path)`` + ``incr()`` reproducer
(``test_fresh_process_per_step_stays_fast`` below) ran in **3.02s on real
Windows CI** — not slow at all. That ruled out generic
``pysciqlop_cache``-connection-open cost and pointed back at something
speasy-specific: ``speasy.core.cache.cache.Cache.__init__`` (not the raw
``pysciqlop_cache.Cache``) does, on *every* open::

    if self.version < cache_version:   # "0.0.0" < "3.0" on a fresh cache
        self._data.clear()
        self.version = cache_version

On a never-before-touched cache directory (the norm on an ephemeral CI
runner), every process sees the stale default version and calls
``clear()`` — which iterates and deletes every file in the cache
directory (see ``_Store::clear()`` in ``store.hpp``). With 4 fresh
processes racing to open the same brand-new cache simultaneously, several
(not just the loser of a lock) can all attempt this delete-and-reinit at
once — a thundering herd, not just a lock wait. Locally reproducing this
exact version-check-then-``clear()`` race with ``spawn`` on Linux stayed
fast (0.60s for 10x4,  see git history / speasy repo notes), so the
`clear()` thundering herd itself isn't inherently slow — it's plausibly
Windows-specific file-deletion contention (``std::filesystem::remove_all``
racing across processes hits Windows sharing-violation retries far more
than POSIX unlink does).

``test_fresh_process_per_step_with_version_check_and_clear`` below
reproduces speasy's exact ``Cache.__init__`` logic (version check +
conditional ``clear()``) directly against ``pysciqlop_cache``, with no
speasy dependency, to confirm this specific mechanism on real Windows CI.

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


def _open_like_speasy_and_incr(path, key):
    """Mirrors speasy.core.cache.cache.Cache.__init__ exactly: version
    check against a hardcoded target, clear() + version bump if stale."""
    cache = Cache(cache_path=path)
    cache.reset_stats()
    version = cache.get("cache/version", "0.0.0")
    if version < "3.0":
        cache.clear()
        cache["cache/version"] = "3.0"
    cache.incr(key, 1, default=0)


class TestWindowsMultiprocessStallRepro(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run_fresh_process_steps(self, target, n_steps, max_seconds_per_step):
        t0 = time.monotonic()
        for step in range(n_steps):
            key = f"counter::{step}"
            processes = [
                Process(target=target, args=(self.tmp_dir, key))
                for _ in range(4)
            ]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            print(f"step {step}: {time.monotonic() - t0:.2f}s elapsed", flush=True)

        elapsed = time.monotonic() - t0
        per_step = elapsed / n_steps
        self.assertLess(
            per_step, max_seconds_per_step,
            f"{n_steps} steps x 4 fresh-process cache opens took "
            f"{elapsed:.1f}s ({per_step:.2f}s/step) - expected under "
            f"{max_seconds_per_step}s/step. This is the exact pattern "
            f"behind the Speasy multiprocess Windows CI stall.",
        )

    def test_fresh_process_per_step_stays_fast(self):
        """Regression guard: N steps of 4 fresh-process opens (bare
        pysciqlop_cache.Cache, no version-check/clear dance) must stay
        fast. Confirmed fast on real Windows CI (3.02s/10 steps,
        2026-07-14) - this is NOT the mechanism behind the Speasy stall."""
        self._run_fresh_process_steps(_open_and_incr, n_steps=10, max_seconds_per_step=2.0)

    def test_fresh_process_per_step_with_version_check_and_clear(self):
        """Reproduces speasy's Cache.__init__ version-check-then-clear()
        dance under 4-way fresh-process contention on a never-before-
        touched cache dir - the leading suspect for the real stall
        mechanism (see module docstring)."""
        self._run_fresh_process_steps(
            _open_like_speasy_and_incr, n_steps=10, max_seconds_per_step=2.0)


if __name__ == "__main__":
    unittest.main()
