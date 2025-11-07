# async_frontier_db_fixed.py
import asyncio
import random
import os
import ray

# ---- Tunables ----
SEED_ITEMS = 25
LLM_LATENCY_S = (0.05, 0.15)     # simulate async I/O
HEAVY_LATENCY_S = (0.10, 0.25)   # simulate compute
BRANCHING_P = 0.01
MAX_CHILDREN = 3
DONE = "__DONE__"

N_LLM   = 2
N_HEAVY = 2

random.seed(42)

# ---------------------------------------------------------------------------
# FrontierDB (async)
# ---------------------------------------------------------------------------
@ray.remote
class FrontierDB:
    """
    Centralized async DB that holds LLM + heavy work queues.
    - Workers await get_*_item(); no polling.
    - When both queues are empty and no in-flight work remains, it
      marks shutdown and enqueues DONE sentinels (one per worker),
      so any waiters wake up and exit cleanly.
    """
    def __init__(self, roots):
        self.q_llm = asyncio.Queue()
        self.q_heavy = asyncio.Queue()
        self.shutdown = False

        self.inflight_llm = 0
        self.inflight_heavy = 0

        # Number of workers to wake on shutdown (set by driver)
        self.n_llm = 0
        self.n_heavy = 0

        # Seed initial roots
        for r in roots:
            self.q_llm.put_nowait(r)

    async def set_worker_counts(self, n_llm: int, n_heavy: int):
        self.n_llm, self.n_heavy = n_llm, n_heavy

    async def get_llm_item(self):
        if self.shutdown:
            return DONE
        # If empty, check for drain; otherwise just await next item
        if self.q_llm.empty():
            await self._maybe_mark_shutdown()
            if self.shutdown:
                return DONE
        item = await self.q_llm.get()
        # If shutdown was signaled by a DONE token, just pass it through
        if item == DONE:
            # Re-enqueue for other waiters then return DONE to caller
            await self.q_llm.put(DONE)
            return DONE
        self.inflight_llm += 1
        return item

    async def get_heavy_item(self):
        if self.shutdown:
            return DONE
        if self.q_heavy.empty():
            await self._maybe_mark_shutdown()
            if self.shutdown:
                return DONE
        item = await self.q_heavy.get()
        if item == DONE:
            await self.q_heavy.put(DONE)
            return DONE
        self.inflight_heavy += 1
        return item

    async def finish_llm(self, children):
        for payload, needs_heavy in children:
            if needs_heavy:
                await self.q_heavy.put(payload)
            else:
                await self.q_llm.put(payload)
        self.inflight_llm -= 1
        await self._maybe_mark_shutdown()

    async def finish_heavy(self, children):
        for payload, needs_heavy in children:
            if needs_heavy:
                await self.q_heavy.put(payload)
            else:
                await self.q_llm.put(payload)
        self.inflight_heavy -= 1
        await self._maybe_mark_shutdown()

    async def _maybe_mark_shutdown(self):
        """If both queues are empty and no work is in flight, broadcast DONE."""
        if self.shutdown:
            return
        empty = self.q_llm.empty() and self.q_heavy.empty()
        idle = self.inflight_llm == 0 and self.inflight_heavy == 0
        if empty and idle:
            self.shutdown = True
            # Wake all waiters: push one DONE per worker (at least 1)
            for _ in range(max(1, self.n_llm)):
                await self.q_llm.put(DONE)
            for _ in range(max(1, self.n_heavy)):
                await self.q_heavy.put(DONE)

# ---------------------------------------------------------------------------
# Workers (async)
# ---------------------------------------------------------------------------
def make_children(item, heavy_prob: float, kmax: int):
    kids = []
    if random.random() < BRANCHING_P:
        for i in range(random.randint(1, kmax)):
            payload = f"{item}/c{i}"
            needs_heavy = random.random() < heavy_prob
            kids.append((payload, needs_heavy))
    return kids

@ray.remote(num_cpus=0.25, max_concurrency=128)
class LLMWorker:
    async def run(self, db):
        while True:
            # ✅ Await the ObjectRef directly; do NOT call .__await__()
            item = await db.get_llm_item.remote()
            print(item)
            if item == DONE:
                return "llm-done"
            # simulate async I/O (e.g., LLM API)
            await asyncio.sleep(random.uniform(*LLM_LATENCY_S))
            children = make_children(item, heavy_prob=0.5, kmax=MAX_CHILDREN)
            await db.finish_llm.remote(children)

@ray.remote(num_cpus=1)
class HeavyWorker:
    async def run(self, db):
        while True:
            item = await db.get_heavy_item.remote()
            print(item)
            if item == DONE:
                return "heavy-done"
            await asyncio.sleep(random.uniform(*HEAVY_LATENCY_S))
            children = make_children(item, heavy_prob=0.3, kmax=MAX_CHILDREN)
            await db.finish_heavy.remote(children)

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
async def main():
    roots = [f"root-{i}" for i in range(SEED_ITEMS)]
    db = FrontierDB.remote(roots)

    llm_workers = [LLMWorker.remote() for _ in range(N_LLM)]
    heavy_workers = [HeavyWorker.remote() for _ in range(N_HEAVY)]

    # Tell DB how many workers exist so it can broadcast DONE properly
    await db.set_worker_counts.remote(N_LLM, N_HEAVY)

    # Start all workers and await their completion
    tasks = [w.run.remote(db) for w in (llm_workers + heavy_workers)]
    # ✅ Just pass the ObjectRefs; Ray makes them awaitable
    results = await asyncio.gather(*tasks)

    print("[done]", results)

if __name__ == "__main__":
    ray.init()
    asyncio.run(main())
    ray.shutdown()