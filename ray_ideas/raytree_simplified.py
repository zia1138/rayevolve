# pool_single_worker_single_queue.py
import time
import random
from typing import List, Tuple
import ray
from ray.util.queue import Queue

# ---- Tunables ----
SEED_ITEMS = 5
LLM_LATENCY_S   = (0.05, 0.15)   # simulate blocking I/O
HEAVY_LATENCY_S = (0.10, 0.25)   # simulate CPU compute
BRANCHING_P = 0.4
MAX_CHILDREN = 5
DONE = "__DONE__"

N_WORKERS = 4  # size of the SingleWorker pool

random.seed(42)

# ---- Helpers ----
def make_children(item: str, heavy_prob: float, kmax: int) -> List[Tuple[str, bool]]:
    """
    Returns a list of (payload, needs_heavy) children.
    """
    kids = []
    if random.random() < BRANCHING_P:
        for i in range(random.randint(1, kmax)):
            payload = f"{item}/c{i}"
            needs_heavy = (random.random() < heavy_prob)
            kids.append((payload, needs_heavy))
    return kids

# ---- Frontier: owns counters + drain detection, queue passed in ----
@ray.remote
class Frontier:
    """
    Frontier tracks a single 'pending' count and uses a Ray Queue
    for work distribution.

    Queue items are (stage, payload) where stage in {"llm", "heavy"}.
    When 'pending' reaches 0, Frontier enqueues one DONE sentinel per worker.
    """
    def __init__(self, roots, work_q, n_workers: int):
        self.q = work_q
        self.n_workers = n_workers
        self.pending = 0
        self.shutdown = False

        # Seed initial work as LLM tasks
        for r in roots:
            self.q.put(("llm", r))
            self.pending += 1

    def submit_children(self, children_with_stage: List[Tuple[str, str]]):
        """
        Called by workers after they finish processing one item.

        children_with_stage: list of (stage, payload),
          where stage in {"llm", "heavy"}.

        We consumed 1 task and added len(children) new tasks:
          pending += len(children) - 1
        """
        if self.shutdown:
            # Ignore new children after shutdown is triggered
            return

        k = len(children_with_stage)
        # One finished, k added
        self.pending += (k - 1)

        # Enqueue children
        for stage, payload in children_with_stage:
            self.q.put((stage, payload))

        # Drain detection: when no more work exists, push DONE sentinels
        if self.pending == 0:
            self.shutdown = True
            for _ in range(self.n_workers):
                self.q.put(DONE)

# ---- SingleWorker: pool of identical workers consuming from the same queue ----
@ray.remote(num_cpus=1)
class SingleWorker:
    def run(self, frontier, work_q: Queue):
        while True:
            item = work_q.get()  # blocks until something arrives
            print("[WORKER]", item)

            if item == DONE:
                return "worker-done"

            stage, payload = item
            if stage == "llm":
                # simulate LLM-ish latency
                time.sleep(random.uniform(*LLM_LATENCY_S))
                children_raw = make_children(payload, heavy_prob=0.5, kmax=MAX_CHILDREN)
            else:
                # stage == "heavy"
                time.sleep(random.uniform(*HEAVY_LATENCY_S))
                children_raw = make_children(payload, heavy_prob=0.3, kmax=MAX_CHILDREN)

            # Map raw children to (stage, payload)
            children_with_stage = []
            for child_payload, needs_heavy in children_raw:
                child_stage = "heavy" if needs_heavy else "llm"
                children_with_stage.append((child_stage, child_payload))

            # Notify frontier that we finished 1 item and spawned children
            frontier.submit_children.remote(children_with_stage)

# ---- Driver ----
def main():
    ray.init()

    roots = [f"root-{i}" for i in range(SEED_ITEMS)]

    # Single shared queue created in the driver
    work_q = Queue()  # you can use Queue(maxsize=...) to add explicit backpressure

    frontier = Frontier.remote(roots=roots, work_q=work_q, n_workers=N_WORKERS)

    workers = [SingleWorker.remote() for _ in range(N_WORKERS)]
    tasks = [w.run.remote(frontier, work_q) for w in workers]

    results = ray.get(tasks)
    print("[done]", results)

    ray.shutdown()

if __name__ == "__main__":
    main()