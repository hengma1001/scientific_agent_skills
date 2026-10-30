---
name: aurora-parsl-academy
description: Run any Parsl + Academy agent workload on ALCF Aurora - choosing a queue, sizing blocks to GPU tiles, the exchange host, keeping the driver alive and stopping it safely, surviving node failures, and finding the real error when a job ends with work missing. Use when writing, submitting, monitoring, debugging, resuming, or killing a Parsl/Academy job on Aurora, whatever the payload (MD, ML inference, analysis).
---

# Parsl + Academy on Aurora

How the pieces fit:

```
login node                         compute nodes (one PBS block)
----------                         ------------------------------
driver.py ── Parsl DFK ── qsub ──> mpiexec --ppn 1  (one manager per node)
    │                                 └─ 12 workers/node, 1 GPU tile each
    └── Academy HTTP exchange <────── each worker hosts one long-lived agent
```

Each agent occupies one Parsl worker slot **for its whole life**, so 1 agent ≈ 1 GPU
tile. Work is sent to agents as action calls through the exchange on the login node.

A working skeleton is in `templates/driver.py`. Copy it, replace the agent.

## 1. Pick the queue (verify with `qstat -Qf <queue>`)

| Queue | Nodes | Max wall | Use for |
|---|---|---|---|
| `debug` | 1-2 | 1 h | smoke tests, benchmarks |
| `debug-scaling` | 1-256 | 1 h | short wide tests |
| `capacity` | 1-16 | 168 h | pilots, resumes, long small jobs |
| `prod` → `small` | **256**-1024 | 12 h | production |
| `prod` → `medium` / `large` | 1,025+ / 2,000+ | 18 / 24 h | very large |

- `prod` **rejects anything under 256 nodes**. `capacity` holds only 16 nodes in
  total and may already be full. `debug` and `debug-scaling` cap walltime at 1 h, and `qsub`
  refuses a longer one ("violates queue and/or server resource limits").
- To finish a few hundred leftover tasks, use `capacity` with a long walltime
  rather than reserving a 256-node `prod` block that will sit mostly idle.
- Waits for 256 nodes have run from hours to two days.

## 2. Size the block

- 12 GPU tiles per node with `ZE_FLAT_DEVICE_HIERARCHY=FLAT`. Use
  `available_accelerators=[str(i) for i in range(12)]` and `max_workers_per_node=12`.
  Parsl then sets `ZE_AFFINITY_MASK` per worker.
- 208 hardware threads per node. 16 cores per worker leaves headroom.
- **Fill the block.** If tasks ≤ workers, wall-clock is one task no matter how
  many tasks there are. You pay for the whole block, so extra tasks that fit are
  free: we ran 5 replicates instead of 3 at no extra cost.
- If tasks > workers, the slowest agent sets the wall-clock: with
  `ceil(tasks/agents)` tasks each, the agents holding the extra task run one task
  longer than the rest.
- **Benchmark at full 12-tile occupancy** before costing a production run. One
  pre-benchmark estimate was 14x too high.

## 3. Walltime

Check that the walltime can finish at least one task, plus ~15% overhead for
startup. A job whose wall is shorter than a single task completes **nothing**:
one 2,488-node job finished 0 of 29,847 tasks this way. Put the check in the
driver and make it refuse to submit (see the template).

## 4. The exchange host — never `localhost`

Agents on compute nodes connect back to the exchange. On a compute node,
`localhost` means that node, so every agent fails with
`Cannot connect to host localhost:<port>`. Bind the exchange to an address that
compute nodes can reach:

| Login-node interface | Reachable from compute nodes? |
|---|---|
| `hsn*` (high-speed network) | **yes** — preferred |
| `bond0` | yes |
| `pub0`, `ib*` | no |

```python
from parsl.addresses import address_by_interface
host = address_by_interface("hsn0")
with spawn_http_exchange(host, port) as factory: ...
```

Use a **different port for each concurrent driver**. Before reusing a port, make
sure nothing else is listening on it (`ss -ltn | grep <port>`). A stray listener
answers the driver and fails as `Connection reset by peer`.

## 5. Keep the driver alive

The driver must stay up for the whole job: Parsl dispatches to agents only once
the block starts. Detach it completely:

```bash
setsid nohup python driver.py -c cfg.yml > run.out 2>&1 < /dev/null & disown
```

A plain `&` inside a tool or terminal session can be killed together with that
session. If you're unsure whether the driver is alive, check that its log is
still growing; `ps` visibility can be limited inside sandboxes.

## 6. Stop a job safely — driver first, then `qdel`

If you `qdel` while the driver is still running, Parsl submits a replacement
block. Kill the driver first:

```bash
kill <driver_pid>; sleep 5; qdel <jobid>
```

**Orphans:** if a driver dies, its queued block stays in the queue. When it
starts, it holds the nodes for the full walltime and does nothing. After any
driver crash, run `qstat -u $USER` and `qdel` anything that has no driver.

When using `pkill -f`, write the pattern so it can't match the shell running
the command, e.g. `pkill -f "drive[r].py"`.

## 7. Node failures take down the whole block

`MpiExecLauncher` starts **one `mpiexec` spanning every node in the block**. If any
single node stops responding, every worker in the block dies with it:

```
ping failed on x4303c3s1b0n0: No reply from x4712c1s0b0n0.hsn... after 108s
→ block exits 127, every agent raises AgentTerminatedError
```

Mitigations:
- `retries=2` in the Parsl `Config`, so lost tasks get dispatched again.
- Smaller blocks shrink the damage: a 16-node block loses less than a 256-node one.
- Make tasks idempotent and resumable, and write checkpoints that a rerun can
  continue from.

## 8. When work goes missing, find the real cause

1. **Driver summary.** Count completions **on disk**, not from action return
   values (see §10).
2. **Block exit status.**
   `qstat -xf <jobid> | grep -E "Exit_status|resources_used.walltime|comment"`.
   Exit 127 together with a `ping failed` message means a node died (§7).
3. **Worker logs.** These are in `<run_dir>/run-info/NNN/submit_scripts/*.stderr`.
   Exceptions raised inside agents show up there, **not** in the driver's log.
   Grep for your failure marker and count the distinct reasons.
4. **Overlap test.** Check which items failed in each replicate or rerun. If the
   failed items differ each time, the cause is infrastructure. If the same items
   fail every time, the cause is something about those items.
5. **Errors at a glance:**

| Symptom | Usual cause |
|---|---|
| `Cannot connect to host localhost` | exchange bound to localhost (§4) |
| `Connection reset by peer` at startup | another process holds the port |
| `AgentTerminatedError: Mailbox ... terminated` | block ended under the agent: node loss, walltime, or a stuck action |
| block runs, 0 tasks done, exit 127 | node failure (§7) |
| `TypeError: argument after ** must be a mapping` | passed a pydantic model as agent kwargs (§10) |
| `terminate called without an active exception` | GPU runtime under `fork`/`spawn` multiprocessing (§9) |

## 9. GPU and environment gotchas

- `module load frameworks` and activate your venv in `worker_init`. Set
  `export ZE_FLAT_DEVICE_HIERARCHY=FLAT` there as well.
- Inside a venv, use `python -m pip`. Plain `pip` can resolve to the frameworks
  module's pip, which installs to `~/.local`.
- Install your own package **editable** (`python -m pip install -e .
  --no-deps --no-build-isolation`) so code changes reach the workers without a
  reinstall. Your home directory and `/lus/flare` are visible from compute nodes.
- **Don't use Python `multiprocessing` with GPU work outside Parsl.** The
  OpenCL / Level Zero runtime aborts under both `fork` and `spawn`. Either let
  Parsl workers own the GPUs, or run CPU-only in a pool (use `spawn` and cap
  threads per process).
- Run heavy CPU preparation (building inputs, post-processing) in a
  compute-node job, not on the shared login node.

## 10. Academy patterns that bite

- **Agent kwargs must be plain data.** `manager.launch(A, kwargs={...})` sends the
  kwargs to a worker. A pydantic model is not a mapping, so `**cfg` inside the
  agent raises `TypeError`. Pass `cfg.model_dump()` instead.
- **Isolate each item inside an action.** Wrap each unit of work in
  `try/except` and `continue` on failure. Otherwise one bad item kills the agent
  and everything still queued behind it.
- **Never let logging raise.** We had a summary log line that referenced a
  renamed attribute. It raised *after* the results were saved, so every task
  logged a failure and every action returned an empty list. Wrap cosmetic log
  lines in `try/except`.
- **Count results on disk.** Treat files written by the agents as the source of
  truth for both progress reports and resume.
- **Blocking work goes through `await self.agent_run_sync(fn, *args)`.** This
  keeps the agent's event loop free to answer the exchange.
- `asyncio.gather(..., return_exceptions=True)` on the action calls, so one
  dead agent doesn't hide the results of the others.

## 11. Make every run resumable

- **Mark completion on disk, and only on success.** Write the marker *after*
  the output is validated. Never write it from an exception handler.
- At startup, the driver filters out work that is already done. Rerunning the
  same config then resumes instead of repeating.
- Use **one output directory per protocol variant.** Otherwise a short run's
  markers make the long run skip those items.
- If the work is grouped (families, classes), interleave it when ordering, so a
  job that runs out of walltime leaves balanced partial coverage.
