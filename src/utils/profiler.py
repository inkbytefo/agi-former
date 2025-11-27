## Developer: inkbytefo
## Modified: 2025-11-27

import time
import os
import psutil
import jax

def profile_training_step(step_fn, params, batch, trace_dir=None, step_name="train_step"):
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    if trace_dir:
        try:
            jax.profiler.start_trace(trace_dir)
        except Exception:
            trace_dir = None
    t0 = time.perf_counter()
    out = step_fn(params, batch)
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    t1 = time.perf_counter()
    if trace_dir:
        try:
            jax.profiler.step(step_name)
            jax.profiler.stop_trace()
        except Exception:
            pass
    rss_after = proc.memory_info().rss
    return {
        "duration_sec": t1 - t0,
        "rss_mb": (rss_after - rss_before) / (1024 * 1024),
    }

