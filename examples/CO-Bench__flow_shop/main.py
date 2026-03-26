"""Initial solution for the Flow Shop Scheduling Problem: NEH (Nawaz-Enscore-Ham)."""


def _compute_makespan(seq, matrix, m):
    """Compute makespan for a job sequence (0-indexed jobs)."""
    n = len(seq)
    if n == 0:
        return 0
    completion = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            proc = matrix[seq[i]][j]
            if i == 0 and j == 0:
                completion[i][j] = proc
            elif i == 0:
                completion[i][j] = completion[i][j - 1] + proc
            elif j == 0:
                completion[i][j] = completion[i - 1][j] + proc
            else:
                completion[i][j] = max(completion[i - 1][j], completion[i][j - 1]) + proc
    return completion[-1][-1]


def solve(instance_id, n, m, matrix, upper_bound, lower_bound):
    """
    Solve the permutation flow shop scheduling problem using the NEH heuristic.

    Args:
        instance_id  : (str) Unique identifier, e.g. "tai20_5_0".
        n            : (int) Number of jobs.
        m            : (int) Number of machines.
        matrix       : (list of lists) matrix[job][machine] processing times.
        upper_bound  : (int) Known upper bound for this instance.
        lower_bound  : (int) Known lower bound for this instance.

    Returns:
        A dict with key "job_sequence" containing a 1-indexed permutation of jobs.
    """
    # NEH: sort jobs by total processing time (descending)
    total_times = [(sum(matrix[j]), j) for j in range(n)]
    total_times.sort(key=lambda x: x[0], reverse=True)

    # Build sequence by inserting each job at the best position
    sequence = [total_times[0][1]]

    for k in range(1, n):
        job = total_times[k][1]
        best_makespan = float("inf")
        best_pos = 0

        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = _compute_makespan(candidate, matrix, m)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos

        sequence.insert(best_pos, job)

    # Convert to 1-indexed
    return {"job_sequence": [j + 1 for j in sequence]}
