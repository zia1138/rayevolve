# centralized_sync_db_single_seed_no_flag.py
import os, time, random
from collections import deque
import ray

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

@ray.remote
class FrontierDB:
    """
    Centralized queues + counters + drain detection.
    Roots are seeded once at init; no further external additions.
    When both queues are empty and no items are in-flight, DB sets shutdown=True.
    Afterwards, get_*_item returns DONE to let workers exit cleanly.
    """
    def __init__(self, roots):
        self.q_llm = deque()
        self.q_heavy = deque()

        # accounting
        self.enq_llm = 0; self.deq_llm = 0; self.inflight_llm = 0
        self.enq_heavy = 0; self.deq_heavy = 0; self.inflight_heavy = 0

        # Seed once
        for x in roots:
            self.q_llm.append(x)
            self.enq_llm += 1

        self.shutdown = False
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
        return None  # temporarily empty

    def finish_llm(self, children):
        for payload, needs_heavy in children:
            if needs_heavy:
                self.q_heavy.append(payload); self.enq_heavy += 1
            else:
                self.q_llm.append(payload);   self.enq_llm += 1
        self.inflight_llm -= 1
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
        for payload, needs_heavy in children:
            if needs_heavy:
                self.q_heavy.append(payload); self.enq_heavy += 1
            else:
                self.q_llm.append(payload);   self.enq_llm += 1
        self.inflight_heavy -= 1
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

# ---- Helpers ----
def make_children(item, heavy_prob: float, kmax: int):
    kids = []
    if random.random() < BRANCHING_P:
        for i in range(random.randint(1, kmax)):
            payload = f"{item}/c{i}"
            needs_heavy = (random.random() < heavy_prob)
            kids.append((payload, needs_heavy))
    return kids

# ---- Workers (synchronous) ----
@ray.remote(num_cpus=0.25)   # I/O-ish LLM step
class LLMWorker:
    def run(self, db):
        backoff = 0.01
        while True:
            item = ray.get(db.get_llm_item.remote())
            print(item)
            if item == DONE:
                return "llm-done"
            if item is None:
                time.sleep(backoff); backoff = min(backoff * 2, 0.2)
                continue
            backoff = 0.01

            # blocking I/O simulation
            time.sleep(random.uniform(*LLM_LATENCY_S))

            children = make_children(item, heavy_prob=0.5, kmax=MAX_CHILDREN)
            db.finish_llm.remote(children)

@ray.remote(num_cpus=1)      # CPU-bound heavy step
class HeavyWorker:
    def run(self, db):
        backoff = 0.01
        while True:
            item = ray.get(db.get_heavy_item.remote())
            print(item)
            if item == DONE:
                return "heavy-done"
            if item is None:
                time.sleep(backoff); backoff = min(backoff * 2, 0.2)
                continue
            backoff = 0.01

            # compute simulation
            time.sleep(random.uniform(*HEAVY_LATENCY_S))

            children = make_children(item, heavy_prob=0.3, kmax=MAX_CHILDREN)
            db.finish_heavy.remote(children)

# ---- Driver ----
def main():
    ray.init()

    roots = [f"root-{i}" for i in range(SEED_ITEMS)]
    db = FrontierDB.remote(roots)  # seed once; DB handles shutdown automatically

    llm_workers   = [LLMWorker.remote() for _ in range(N_LLM)]
    heavy_workers = [HeavyWorker.remote() for _ in range(N_HEAVY)]
    tasks = [w.run.remote(db) for w in (llm_workers + heavy_workers)]

    # Wait for graceful drain
    results = ray.get(tasks)
    print("[done]", results)
    
    ray.shutdown()

if __name__ == "__main__":
    main()