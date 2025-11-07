# centralized_sync_db_with_semaphores.py
import os, time, random
from collections import deque
import ray
import asyncio

# ---- Tunables ----
SEED_ITEMS = 25
LLM_LATENCY_S = (0.05, 0.15)   # simulate blocking I/O
HEAVY_LATENCY_S = (0.10, 0.25) # simulate CPU compute
BRANCHING_P = 0.01
MAX_CHILDREN = 3
DONE = "__DONE__"

N_LLM   = 1
N_HEAVY = 1

random.seed(42)

# ---------------- Permit (async semaphore; bounds wakeups) ----------------
@ray.remote
class Permit:
    """
    A per-queue semaphore:
      - Each available item corresponds to one permit.
      - Workers acquire() a permit before asking DB for an item.
      - DB release(k) when it enqueues k items.
      - shutdown() wakes everybody and causes acquire() to yield DONE.
    """
    def __init__(self):
        self._sem = asyncio.Semaphore(0)
        self._shutdown = False

    async def acquire(self):
        # Fast path: if shutdown already requested
        if self._shutdown:
            return DONE
        await self._sem.acquire()
        # If shutdown flipped while waiting, indicate termination
        if self._shutdown:
            return DONE
        return True

    def release(self, n: int = 1):
        # Grant n permits (wake up to n waiters)
        for _ in range(n):
            self._sem.release()

    def shutdown(self):
        # Flip the flag and massively over-release so all waiters wake.
        # (Simplest robust approach; avoids tracking exact waiter count.)
        self._shutdown = True
        for _ in range(100_000):
            self._sem.release()

# ---------------- FrontierDB (sync, centralized) ----------------
@ray.remote
class FrontierDB:
    """
    Centralized queues + counters + drain detection (synchronous).
    Uses per-queue Permits to wake exactly as many workers as new items.
    """
    def __init__(self, roots, permit_llm, permit_heavy):
        self.q_llm = deque()
        self.q_heavy = deque()

        # accounting
        self.enq_llm = 0; self.deq_llm = 0; self.inflight_llm = 0
        self.enq_heavy = 0; self.deq_heavy = 0; self.inflight_heavy = 0

        self.shutdown = False
        self.permit_llm = permit_llm
        self.permit_heavy = permit_heavy

        # Seed once (all roots start in LLM queue)
        for x in roots:
            self.q_llm.append(x)
            self.enq_llm += 1

        # Release permits for the seeded items so exactly that many workers wake
        if self.enq_llm:
            self.permit_llm.release.remote(self.enq_llm)

        # Evaluate initial state
        self._maybe_mark_shutdown()

    # ---- LLM stage ----
    def get_llm_item(self):
        if self.shutdown:
            return DONE
        if self.q_llm:
            item = self.q_llm.popleft()
            self.deq_llm += 1
            self.inflight_llm += 1
            return item
        # With permits, this should be rare; keep as defensive fallback
        return None

    def finish_llm(self, children):
        enq_llm, enq_heavy = 0, 0
        for payload, needs_heavy in children:
            if needs_heavy:
                self.q_heavy.append(payload); self.enq_heavy += 1; enq_heavy += 1
            else:
                self.q_llm.append(payload);   self.enq_llm   += 1; enq_llm   += 1

        self.inflight_llm -= 1

        # Release permits to match new items
        if enq_llm:
            self.permit_llm.release.remote(enq_llm)
        if enq_heavy:
            self.permit_heavy.release.remote(enq_heavy)

        self._maybe_mark_shutdown()

    # ---- Heavy stage ----
    def get_heavy_item(self):
        if self.shutdown:
            return DONE
        if self.q_heavy:
            item = self.q_heavy.popleft()
            self.deq_heavy += 1
            self.inflight_heavy += 1
            return item
        return None

    def finish_heavy(self, children):
        enq_llm, enq_heavy = 0, 0
        for payload, needs_heavy in children:
            if needs_heavy:
                self.q_heavy.append(payload); self.enq_heavy += 1; enq_heavy += 1
            else:
                self.q_llm.append(payload);   self.enq_llm   += 1; enq_llm   += 1

        self.inflight_heavy -= 1

        if enq_llm:
            self.permit_llm.release.remote(enq_llm)
        if enq_heavy:
            self.permit_heavy.release.remote(enq_heavy)

        self._maybe_mark_shutdown()

    # ---- Drain detection ----
    def _maybe_mark_shutdown(self):
        if self.shutdown:
            return
        q_llm_size   = self.enq_llm   - self.deq_llm
        q_heavy_size = self.enq_heavy - self.deq_heavy
        idle  = (self.inflight_llm == 0 and self.inflight_heavy == 0)
        empty = (q_llm_size == 0 and q_heavy_size == 0)
        if idle and empty:
            self.shutdown = True
            # Wake all waiters so they can observe DONE and exit
            self.permit_llm.shutdown.remote()
            self.permit_heavy.shutdown.remote()

# ---- Helpers ----
def make_children(item, heavy_prob: float, kmax: int):
    kids = []
    if random.random() < BRANCHING_P:
        for i in range(random.randint(1, kmax)):
            payload = f"{item}/c{i}"
            needs_heavy = (random.random() < heavy_prob)
            kids.append((payload, needs_heavy))
    return kids

# ---- Workers (no polling; permit-first) ----
@ray.remote(num_cpus=0.25)   # I/O-ish LLM step
class LLMWorker:
    def run(self, db, permit_llm):
        while True:
            tok = ray.get(permit_llm.acquire.remote())
            if tok == DONE:
                return "llm-done"

            item = ray.get(db.get_llm_item.remote())
            print("[LLM]", item)
            if item == DONE:
                return "llm-done"
            if item is None:
                # Defensive fallback (permits should prevent this)
                continue

            # blocking I/O simulation
            time.sleep(random.uniform(*LLM_LATENCY_S))

            children = make_children(item, heavy_prob=0.5, kmax=MAX_CHILDREN)
            ray.get(db.finish_llm.remote(children))

@ray.remote(num_cpus=1)      # CPU-bound heavy step
class HeavyWorker:
    def run(self, db, permit_heavy):
        while True:
            tok = ray.get(permit_heavy.acquire.remote())
            if tok == DONE:
                return "heavy-done"

            item = ray.get(db.get_heavy_item.remote())
            print("[HEAVY]", item)
            if item == DONE:
                return "heavy-done"
            if item is None:
                continue

            # compute simulation
            time.sleep(random.uniform(*HEAVY_LATENCY_S))

            children = make_children(item, heavy_prob=0.3, kmax=MAX_CHILDREN)
            ray.get(db.finish_heavy.remote(children))

# ---- Driver ----
def main():
    ray.init()

    roots = [f"root-{i}" for i in range(SEED_ITEMS)]
    permit_llm   = Permit.remote()
    permit_heavy = Permit.remote()
    db = FrontierDB.remote(roots, permit_llm, permit_heavy)

    llm_workers   = [LLMWorker.remote() for _ in range(N_LLM)]
    heavy_workers = [HeavyWorker.remote() for _ in range(N_HEAVY)]
    tasks = [w.run.remote(db, permit_llm) for w in llm_workers] + \
            [w.run.remote(db, permit_heavy) for w in heavy_workers]

    results = ray.get(tasks)
    print("[done]", results)
    ray.shutdown()

if __name__ == "__main__":
    main()