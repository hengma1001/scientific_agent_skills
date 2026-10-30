"""Skeleton Parsl + Academy driver for Aurora. Copy, then replace WorkAgent.work_one.

Encodes the rules from the aurora-parsl-academy skill: routable exchange host,
walltime check, resumable on-disk completion, per-item isolation, one agent per
GPU tile, results counted from disk.

    python driver.py --items items.txt --out out/ --nodes 2 --queue debug --walltime 1:00:00 --dry-run
    setsid nohup python driver.py ... > run.out 2>&1 < /dev/null & disown
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
from pathlib import Path

from academy.agent import Agent, action
from academy.exchange.cloud import spawn_http_exchange
from academy.manager import Manager
from parsl.addresses import address_by_hostname, address_by_interface
from parsl.concurrent import ParslPoolExecutor
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.providers import PBSProProvider

TILES_PER_NODE = 12
VENV = "/path/to/your/venv"           # <-- edit
ACCOUNT = "YOUR_ALLOCATION"           # <-- edit


# --------------------------------------------------------------------------- agent
class WorkAgent(Agent):
    """One per GPU tile. Processes a list of items, isolating each."""

    def __init__(self, out_dir: str, params: dict) -> None:
        super().__init__()
        self.out_dir = Path(out_dir)
        self.params = params            # plain dict, never a pydantic model

    def work_one(self, item: str) -> dict:
        """Blocking work for one item. Replace this."""
        return {"item": item, "result": len(item)}

    def _run_and_save(self, item: str) -> None:
        result = self.work_one(item)
        # Completion marker written only after the result exists.
        (self.out_dir / f"{item}.json").write_text(json.dumps(result))

    @action
    async def process(self, items: list[str]) -> int:
        done = 0
        for item in items:
            try:
                await self.agent_run_sync(self._run_and_save, item)
                done += 1
            except Exception:  # noqa: BLE001 - one bad item, not the batch
                logging.exception(f"FAILED: {item}")
        return done


# ------------------------------------------------------------------------ helpers
def is_done(item: str, out_dir: Path) -> bool:
    return (out_dir / f"{item}.json").exists()


def exchange_host(interfaces=("hsn0", "bond0")) -> str:
    """Address reachable from compute nodes. Never localhost."""
    for iface in interfaces:
        try:
            addr = address_by_interface(iface)
            if addr:
                return addr
        except Exception:  # noqa: BLE001 - interface absent
            continue
    return address_by_hostname()


def check_walltime(walltime: str, hours_per_item: float, force: bool) -> None:
    h, m, s = ([int(x) for x in walltime.split(":")] + [0, 0])[:3]
    wall = h + m / 60 + s / 3600
    needed = hours_per_item * 1.15 + 0.1
    if wall < needed and not force:
        raise SystemExit(f"walltime {walltime} < ~{needed:.1f} h for one item; nothing would finish")


def parsl_config(args, run_dir: Path) -> Config:
    return Config(
        run_dir=str(run_dir),
        retries=args.retries,           # survive a lost node (whole-block mpiexec)
        executors=[
            HighThroughputExecutor(
                label="htex",
                available_accelerators=[str(i) for i in range(TILES_PER_NODE)],
                max_workers_per_node=TILES_PER_NODE,
                cores_per_worker=16,
                prefetch_capacity=0,
                heartbeat_period=30,
                heartbeat_threshold=300,
                provider=PBSProProvider(
                    launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=208 --ppn 1"),
                    account=ACCOUNT,
                    queue=args.queue,
                    walltime=args.walltime,
                    nodes_per_block=args.nodes,
                    scheduler_options="#PBS -l filesystems=home:flare",
                    worker_init=(
                        f"module load frameworks; source {VENV}/bin/activate; "
                        "export TMPDIR=/tmp; export ZE_FLAT_DEVICE_HIERARCHY=FLAT"
                    ),
                ),
            ),
        ],
    )


# --------------------------------------------------------------------------- main
async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--items", required=True, help="file, one item per line")
    p.add_argument("--out", required=True)
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--queue", default="debug")
    p.add_argument("--walltime", default="1:00:00")
    p.add_argument("--hours-per-item", type=float, default=0.1, help="measured, not guessed")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--port", type=int, default=5500, help="unique per concurrent driver")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "driver.log"), logging.StreamHandler()])

    check_walltime(args.walltime, args.hours_per_item, args.force)

    items = [x.strip() for x in Path(args.items).read_text().splitlines() if x.strip()]
    pending = [x for x in items if not is_done(x, out)]
    n_agents = min(args.nodes * TILES_PER_NODE, len(pending))
    host = exchange_host()
    logging.info(f"items={len(items)} pending={len(pending)} agents={n_agents} exchange={host}:{args.port}")
    if not pending or args.dry_run:
        return

    executor = ParslPoolExecutor(parsl_config(args, out / "run-info"))

    def _sigterm(*_):                 # kill driver -> tear down workers, then exit
        executor.shutdown(wait=False)
        os._exit(0)
    signal.signal(signal.SIGTERM, _sigterm)

    with spawn_http_exchange(host, args.port) as factory:
        try:
            async with await Manager.from_exchange_factory(factory=factory, executors=executor) as manager:
                agents = await asyncio.gather(*[
                    manager.launch(WorkAgent, kwargs={"out_dir": str(out), "params": {}})
                    for _ in range(n_agents)
                ])
                results = await asyncio.gather(
                    *[a.process(pending[i::n_agents]) for i, a in enumerate(agents)],
                    return_exceptions=True,
                )
                done = sum(is_done(x, out) for x in pending)       # truth is on disk
                dead = [r for r in results if isinstance(r, BaseException)]
                logging.info(f"finished: {done}/{len(pending)} done; {len(dead)} agents raised")
                for e in dead[:5]:
                    logging.error(f"agent error: {e!r}")
        finally:
            executor.shutdown(wait=True)


if __name__ == "__main__":
    asyncio.run(main())
