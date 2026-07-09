"""Environment-based preprocessing pipeline.

Loads the elevation and depositional-environment cubes, smooths measurement
noise, extracts per-layer stratigraphy and statistics, drops empty locations, and
writes the tagged layer-stats CSV consumed by the trainers.

Only the final CSV is written; the per-location intermediates stay in memory.
``--workers N`` processes contiguous location-chunks in parallel.
"""

import argparse
import io
import time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

import numpy as np
import pandas as pd
import scipy.io

from stratml import config
from stratml.preprocessing.flatten import flatten
from stratml.preprocessing.error_check import error_check
from stratml.preprocessing.strat import current_strat
from stratml.preprocessing.layer_stats import layer_info
from stratml.preprocessing.tag import tag

ELEV_MAT = config.DATA_ROOT / "ZdataD_final.mat"      # key 'ZD'
ENV_MAT = config.DATA_ROOT / "Depo_Env.mat"           # key 'DE'


def load_mat_data(file, key):
    """Load a MATLAB ``.mat`` file and return ``mat[key]`` as a NumPy array."""
    return np.array(scipy.io.loadmat(file)[key])


def _csv_roundtrip(df, index=False):
    """Round-trip a frame through an in-memory CSV.

    The write/read perturbs float64 at ~1e-14, which can flip a ``High_Erosion``
    value on the 6.5 boundary; applying it at each stage keeps results
    deterministic without writing the multi-GB tensors to disk.
    """
    buf = io.StringIO()
    df.to_csv(buf, index=index)
    buf.seek(0)
    return pd.read_csv(buf)


def _process_chunk(data_env):
    """Run the per-location pipeline on one chunk; return its layer rows.

    Locations are independent. Module-level and picklable so it can run in a
    worker process.
    """
    data_chunk, env_chunk = data_env
    # current_strat needs 0-based row labels and integer columns
    data_chunk = data_chunk.reset_index(drop=True)
    data_chunk.columns = data_chunk.columns.astype(int)
    env_chunk = env_chunk.reset_index(drop=True)

    # smooth measurement noise, then extract per-layer stratigraphy
    data_chunk = error_check(data_chunk)
    data_chunk = _csv_roundtrip(data_chunk)
    data_chunk.columns = data_chunk.columns.astype(int)
    strat, times, start_times, env_chunk = current_strat(data_chunk, env_chunk)

    # empty chunk (no deposition) -> no columns, no layers
    if strat.shape[1] == 0:
        return None

    # round-trip each frame through CSV (see _csv_roundtrip)
    strat = _csv_roundtrip(strat)
    times = _csv_roundtrip(times)
    start_times = _csv_roundtrip(start_times)
    env_chunk = _csv_roundtrip(env_chunk)
    data_chunk = _csv_roundtrip(data_chunk)

    # drop empty (all-zero) locations, keeping every frame aligned
    non_zero = strat.index[strat.sum(axis=1) != 0]
    strat = strat.loc[non_zero]
    times = times.loc[non_zero]
    start_times = start_times.loc[non_zero]
    data_chunk = data_chunk.loc[non_zero]
    env_chunk = env_chunk.loc[non_zero]

    # derive per-layer statistics; keep only real (non-zero thickness) layers
    df = layer_info(strat, times, start_times, data_chunk, env_chunk)
    return df[df['Layer_Thickness'] != 0]


def _process_bounds(data, env, workers, chunk_size):
    """Process fixed-size contiguous chunks; return results in global order.

    Chunk size is decoupled from worker count (small chunks cap per-worker memory),
    and only ~2*workers chunks are kept in flight so peak memory stays flat.
    """
    # contiguous location ranges
    bounds = [(a, min(a + chunk_size, len(data)))
              for a in range(0, len(data), chunk_size)]
    parts = [None] * len(bounds)

    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            in_flight = {}
            nxt = 0
            while nxt < len(bounds) or in_flight:
                # keep at most ~2*workers chunks submitted at once
                while nxt < len(bounds) and len(in_flight) < 2 * workers:
                    a, b = bounds[nxt]
                    fut = pool.submit(_process_chunk, (data.iloc[a:b], env.iloc[a:b]))
                    in_flight[fut] = nxt
                    nxt += 1
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    parts[in_flight.pop(fut)] = fut.result()
    else:
        for i, (a, b) in enumerate(bounds):
            parts[i] = _process_chunk((data.iloc[a:b], env.iloc[a:b]))

    return parts, len(bounds)


def main(workers=1, chunk_size=4000):
    # load and flatten both cubes to (location, timestep) tables
    data = flatten(load_mat_data(ELEV_MAT, 'ZD'))
    env = flatten(load_mat_data(ENV_MAT, 'DE'))

    # process location-chunks and stitch the layer rows back together in order
    start = time.time()
    parts, n_chunks = _process_bounds(data, env, workers, chunk_size)
    df = pd.concat([p for p in parts if p is not None], ignore_index=True)
    print(f"Kept {len(df)} layers "
          f"({n_chunks} chunks of {chunk_size} locs, {workers} worker(s), "
          f"{time.time()-start:.2f}s)")

    # carry the row index into the CSV as the 'Unnamed: 0' column
    df = _csv_roundtrip(df, index=True)
    df = df[df['Layer_Thickness'] != 0]

    # add the High_Erosion tag and write the final dataset
    df_tagged = tag(df)
    config.LAYER_STATS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_tagged.to_csv(config.LAYER_STATS_CSV, index=False)
    print(f"Wrote {config.LAYER_STATS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="parallel location-chunk workers (1 = single process)")
    parser.add_argument(
        "--chunk-size", type=int, default=4000,
        help="locations per chunk; smaller = lower peak memory (default 4000)")
    args = parser.parse_args()
    main(args.workers, args.chunk_size)
