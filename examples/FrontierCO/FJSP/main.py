"""Initial solution for the Flexible Job Shop Scheduling Problem (FJSP)."""


def solve(instance_id, num_jobs, num_machines, jobs):
    """
    Solves the Flexible Job Shop Scheduling Problem.

    Args:
        instance_id  : (str) Unique identifier for this problem instance.
        num_jobs     : (int) Number of jobs.
        num_machines : (int) Number of machines.
        jobs         : (list) A list of jobs, where each job is a list of operations.
                       Each operation is a list of (machine, time) pairs representing
                       compatible machines and their processing times.
                       Note: Items are always 1-indexed.

    Returns:
        A dict with keys:
            - "makespan" (float): The completion time of the last operation.
            - "machine_assignments" (list): Machine assigned to each operation (globally indexed).
            - "start_times" (list): Start time of each operation (globally indexed).
    """
    # Simple greedy: assign each operation to its first compatible machine, schedule sequentially
    machine_assignments = []
    start_times = []
    machine_end = {}  # machine -> earliest available time
    makespan = 0.0

    for job in jobs:
        job_time = 0.0
        for operation in job:
            # Pick the first compatible machine
            chosen_machine, processing_time = operation[0]
            start = max(job_time, machine_end.get(chosen_machine, 0.0))
            machine_assignments.append(chosen_machine)
            start_times.append(start)
            end = start + processing_time
            machine_end[chosen_machine] = end
            job_time = end
            if end > makespan:
                makespan = end

    return {
        "makespan": makespan,
        "machine_assignments": machine_assignments,
        "start_times": start_times,
    }
