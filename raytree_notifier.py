# centralized_sync_db_with_notifier.py
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

# ---------------- Notifier ----------------
@ray.remote
class Notifier:
    """
    A tiny event primitive:
      - wait(): returns an ObjectRef token; ray.get(token) blocks until next signal().
      - signal(): rotates the token so all current waiters wake up.
    """
    def __init__(self):
        self._token = ray.put(object())  # current epoch token

    def wait(self):
        return self._token

    def signal(self):
        self._token = ray.put(object())

# ---------------- FrontierDB (sync) ----------------
@ray.remote
class FrontierDB:
    """
    Centralized queues + counters + drain detection (synchronous).
    Uses Notifier to wake workers when new work arrives or when shutdown occurs.
    """
    def __init__(self, roots, notifier):
        self.q_llm = deque()
        self.q_heavy = deque()

        # accounting
        self.enq_llm = 0; self.deq_llm = 0; self.inflight_llm = 0
        self.enq_heavy = 0; self.deq_heavy = 0; self.inflight_heavy = 0

        self.shutdown = False
        self.notifier = notifier

        # Seed once
        for x in roots:
            self.q_llm.append(x)
            self.enq_llm += 1

        # After seeding, we might already be empty (degenerate case). Check & signal.
        self._maybe_mark_shutdown()
        # Either way, wake initial waiters so they start pulling.
        self.notifier.signal.remote()

    # ---- LLM stage ----
    def get_llm_item(self):
        if self.shutdown:
            return DONE
        if self.q_llm:
            item = self.q_llm.popleft()
            self.deq_llm += 1
            self.inflight_llm += 1
            return item
        return None  # temporarily empty; worker should wait on notifier

    def finish_llm(self, children):
        # Route children
        enq_count = 0
        for payload, needs_heavy in children:
            if needs_heavy:
                self.q_heavy.append(payload); self.enq_heavy += 1
            else:
                self.q_llm.append(payload);   self.enq_llm += 1
            enq_count += 1

        # Mark completion
        self.inflight_llm -= 1

        # If we enqueued anything or changed state, wake waiters
        if enq_count > 0:
            self.notifier.signal.remote()

        # Possibly transition to shutdown
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
        enq_count = 0
        for payload, needs_heavy in children:
            if needs_heavy:
                self.q_heavy.append(payload); self.enq_heavy += 1
            else:
                self.q_llm.append(payload);   self.enq_llm += 1
            enq_count += 1

        self.inflight_heavy -= 1

        if enq_count > 0:
            self.notifier.signal.remote()

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
            self.notifier.signal.remote()

# ---- Helpers ----
def make_children(item, heavy_prob: float, kmax: int):
    kids = []
    if random.random() < BRANCHING_P:
        for i in range(random.randint(1, kmax)):
            payload = f"{item}/c{i}"
            needs_heavy = (random.random() < heavy_prob)
            kids.append((payload, needs_heavy))
    return kids

# ---- Workers (synchronous, no polling) ----
@ray.remote(num_cpus=0.25)   # I/O-ish LLM step
class LLMWorker:
    def run(self, db, notifier):
        while True:
            item = ray.get(db.get_llm_item.remote())
            if item == DONE:
                return "llm-done"
            if item is None:
                # Block until DB signals new work *or* shutdown
                ray.get(notifier.wait.remote())
                continue

            # blocking I/O simulation
            time.sleep(random.uniform(*LLM_LATENCY_S))

            children = make_children(item, heavy_prob=0.5, kmax=MAX_CHILDREN)
            ray.get(db.finish_llm.remote(children))

@ray.remote(num_cpus=1)      # CPU-bound heavy step
class HeavyWorker:
    def run(self, db, notifier):
        while True:
            item = ray.get(db.get_heavy_item.remote())
            if item == DONE:
                return "heavy-done"
            if item is None:
                ray.get(notifier.wait.remote())
                continue

            # compute simulation
            time.sleep(random.uniform(*HEAVY_LATENCY_S))

            children = make_children(item, heavy_prob=0.3, kmax=MAX_CHILDREN)
            ray.get(db.finish_heavy.remote(children))

# ---- Driver ----
def main():
    ray.init()

    roots = [f"root-{i}" for i in range(SEED_ITEMS)]
    notifier = Notifier.remote()
    db = FrontierDB.remote(roots, notifier)  # seed once; DB handles shutdown + signals

    llm_workers   = [LLMWorker.remote() for _ in range(N_LLM)]
    heavy_workers = [HeavyWorker.remote() for _ in range(N_HEAVY)]
    tasks = [w.run.remote(db, notifier) for w in (llm_workers + heavy_workers)]

    results = ray.get(tasks)
    print("[done]", results)

    ray.shutdown()

if __name__ == "__main__":
    main()