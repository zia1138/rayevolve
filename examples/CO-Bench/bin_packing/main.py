"""Seed heuristic for 1D bin packing: First Fit Decreasing (FFD)."""


def solve(**kwargs):
    """
    Solve the one-dimensional bin packing problem.

    Input kwargs:
      - id:           Problem identifier (string)
      - bin_capacity: Capacity of each bin (float)
      - num_items:    Number of items (int)
      - items:        List of item sizes (list of floats)

    Returns:
      dict with:
        - 'num_bins': Number of bins used (int)
        - 'bins': List of lists, each containing 1-based item indices
    """
    bin_capacity = kwargs["bin_capacity"]
    num_items = kwargs["num_items"]
    items = kwargs["items"]

    # Create list of (size, original_1based_index) and sort descending by size
    indexed_items = [(items[i], i + 1) for i in range(num_items)]
    indexed_items.sort(key=lambda x: x[0], reverse=True)

    bins = []          # list of lists of 1-based item indices
    bin_remaining = []  # remaining capacity in each bin

    for size, idx in indexed_items:
        # First Fit: find the first bin that can accommodate this item
        placed = False
        for b in range(len(bins)):
            if bin_remaining[b] >= size:
                bins[b].append(idx)
                bin_remaining[b] -= size
                placed = True
                break

        if not placed:
            # Open a new bin
            bins.append([idx])
            bin_remaining.append(bin_capacity - size)

    return {
        "num_bins": len(bins),
        "bins": bins,
    }
