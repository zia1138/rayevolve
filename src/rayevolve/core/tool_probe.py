"""Probe utilities for rayevolve.

Extracted from worker2.py to keep the core worker focused on evolution logic.
Provides:
- extract_probe_summary(): Extract actionable findings from probe stdout
- build_probe_docstring(): Build the full probe tool docstring
- DEFAULT_EXTRA_PROBE_INSTRUCTIONS: Generic probe guidance (domain-agnostic)
"""


def extract_probe_summary(stdout: str, max_len: int = 1500) -> str:
    """Extract actionable findings from probe stdout for cross-generation memory.

    Extraction priority:
    1. '### Summary' section -- best case, probe followed instructions
    2. Content after '====== PROBE RESULTS ======' -- strips preprocessing boilerplate
    3. Tail of stdout -- last resort
    """
    if not stdout or not stdout.strip():
        return ""
    marker = "### Summary"
    idx = stdout.rfind(marker)
    if idx >= 0:
        summary = stdout[idx + len(marker):].strip()
        return summary[:max_len]
    probe_marker = "====== PROBE RESULTS ======"
    idx = stdout.find(probe_marker)
    if idx >= 0:
        after = stdout[idx + len(probe_marker):]
        safety_marker = "====== FROZEN SAFETY CHECKS ======"
        safety_idx = after.find(safety_marker)
        if safety_idx >= 0:
            after = after[:safety_idx]
        after = after.strip()
        if not after:
            return ""
        if after.startswith("Evaluation error:") and "\n" not in after.strip():
            return ""
        return after[:max_len]
    stdout = stdout.strip()
    if len(stdout) > max_len:
        return "...[truncated]...\n" + stdout[-max_len:]
    return stdout


# ---------------------------------------------------------------------------
# Generic probe instructions -- the universal contract for probe tools.
# Application-specific guidance (hooks, pitfalls, output formatting) should
# go in EvolutionConfig.extra_probe_instructions or be passed to
# build_probe_docstring() at runtime.
# ---------------------------------------------------------------------------
_GENERIC_PROBE_INSTRUCTIONS_BODY = """\
Run a diagnostic probe: modify the program to emit targeted,
low-cost evidence about why it is performing as it is.

A probe can investigate any actionable structure, including:
- relationships between quantities (correlations, interactions,
  conditional behavior, regime changes),
- systematic failure modes or error clusters,
- sensitivity to parameters or hyperparameters,
- surprising invariants or inconsistencies.

Your intent should identify the specific quantity or relationship
to measure. Structure your probe code to directly answer it.

Probe design:
1. Identify the quantity that answers your intent.
2. Intercept the program's execution to collect that quantity --
   override an existing function and accumulate values, or define
   a hook function.
3. After all runs complete, aggregate and rank-order components.
4. Print a concise comparison table, then a ### Summary with
   findings.

Quality bar:
- Do NOT use probes for basic correctness checks ("does it run?").
- Go beyond shape/head printing -- report at least one
  relationship-level finding (dependency, pattern, anomaly).
- Structure your analysis to directly answer the stated intent.

How your code executes:
- Your code is APPENDED after the current program. You inherit all
  existing function definitions, variables, and constants.
- Your code runs at module level, before the evaluation harness
  calls the program's entry points.
- You may define new helper functions, but you MUST also call them
  or they will never execute.
- To inject analysis into the execution flow, OVERRIDE an existing
  function using the EXACT SAME NAME. Save the original first:
      _saved = existing_func
      def existing_func(...):
          result = _saved(...)  # call original
          # ... your analysis / prints ...
          return result  # MUST return the original result
- NEVER use `import __main__; __main__.func = new_func`.
- NEVER return a different value from the override.

Output formatting:
- IMPORTANT: End your output by printing a "### Summary" header
  followed by 2-6 bullet points. This section is extracted for
  cross-generation memory, so it must be SELF-CONTAINED and
  DECISION-ORIENTED."""

_GENERIC_PROBE_ARGS = """
Args:
    probe_code: Modified program with diagnostic instrumentation.
    intent: The hypothesis/question being tested and what decision it informs.
Returns:
    stdout/stderr from running the probe."""

# Composed constant for backward compatibility (tests, imports)
_GENERIC_PROBE_INSTRUCTIONS = _GENERIC_PROBE_INSTRUCTIONS_BODY + "\n" + _GENERIC_PROBE_ARGS


# ---------------------------------------------------------------------------
# Default extra probe instructions -- domain-agnostic guidance that projects
# can override via EvolutionConfig.extra_probe_instructions.
# ---------------------------------------------------------------------------
DEFAULT_EXTRA_PROBE_INSTRUCTIONS = """\
Most common probe patterns:
- Component relationships: pairwise correlations between outputs
  to find redundancy or independence.
- Component ranking: order by a metric (score, coefficient,
  separability) to identify strongest/weakest.
- Distribution diagnostics: scale, variance, skew of outputs to
  detect imbalance or outliers.
- Processing bottlenecks: hotspots in time, memory, or IO that
  limit throughput or cause timeouts.

Accessing internal state -- choose the right strategy:

  (A) Hook function already exists in the evolved code:
      Some programs define a dedicated hook function (e.g.,
      `probe_hook(ctx)`) that is called at a strategic point --
      typically after all processing is complete -- and receives
      aggregated internal state as a dict. If such a hook exists,
      OVERRIDE it to access the data you need. Check the system
      message or the evolved code for the hook's signature and
      available context keys.
          _original_hook = probe_hook
          def probe_hook(ctx):
              _original_hook(ctx)  # preserve original behavior
              # ... your analysis using ctx ...

  (B) No hook function -- intercept existing functions:
      When no hook exists, override one or more functions in the
      evolved code to intercept execution. This works well for
      programs that call a function once (e.g., an optimization
      routine, a constructor, a scoring function). Save the
      original, call it inside your override, analyze the result,
      then return it unchanged.
          _saved_fn = some_function
          def some_function(*args, **kwargs):
              result = _saved_fn(*args, **kwargs)
              # ... your analysis of result ...
              return result  # MUST return original result

  (C) Design a new hook when neither (A) nor (B) suffices:
      If the data you need is buried deep inside the program and
      no single function exposes it, you may inject a new hook
      point. Override the innermost function that touches the data,
      accumulate values into a module-level list, and print your
      analysis after all calls complete. Use `_IS_PROBE` (set
      automatically) to guard probe-only code paths so they don't
      affect the score when not probing.
          _probe_data = []
          _saved_inner = inner_function
          def inner_function(*args, **kwargs):
              result = _saved_inner(*args, **kwargs)
              if _IS_PROBE:
                  _probe_data.append(extract_what_you_need(result))
              return result
          # At module level, AFTER all definitions:
          # (this runs when the program executes)
          # ... call the entry point, then analyze _probe_data ...

Output formatting (extended):
- If your override runs multiple times (e.g., once per iteration
  or fold), collect values into a global list and print a single
  aggregated summary at the end (never at module level before
  execution completes).
- Rank-order results so the most important items appear first.
- Flag instabilities: if a value changes sign across runs or has
  high variance relative to its mean, note it explicitly.
- Before writing each Summary bullet, RANK all components by the
  metric you measured and assign verdicts based on RELATIVE
  position, not absolute value alone. A positive value does NOT
  mean KEEP -- if it's 10x smaller than the best, it may warrant
  REMOVE or INVESTIGATE.

  Ranking rules:
  - Show the TOP 2-3 names+values in every ranking, not just #1.
  - Use DOMINANT only when #1 is >2x #2. Otherwise use NEAR-TIE
    and list the cluster (e.g., "A (0.13) = B (0.13) = C (0.11)").
  - Show TIER STRUCTURE when it exists (e.g., "top 3 at 0.11-0.13,
    bottom 2 at 0.05-0.06").

  Each bullet must:
  (a) State a concrete verdict: KEEP/REMOVE/INVESTIGATE + why,
      justified by comparison to the strongest component
  (b) Include the key value AND the comparison baseline
  (c) Use f-strings with computed variables -- never hardcode
  If the intent asked multiple questions and the probe only
  answered some, add a bullet: "NOT MEASURED: <what's missing>".
  Example:
      # Sort components by metric FIRST, then assign verdicts
      ranked = sorted(zip(names, coefs), key=lambda x: abs(x[1]), reverse=True)
      best_name, best_val = ranked[0]
      sec_name, sec_val = ranked[1]
      worst_name, worst_val = ranked[-1]
      ratio_1_2 = abs(best_val / sec_val) if sec_val != 0 else float('inf')
      print("### Summary")
      # Use DOMINANT only when >2x gap; otherwise NEAR-TIE with top 2-3
      if ratio_1_2 > 2:
          print(f"- DOMINANT: {best_name} (coef={best_val:.3f}) -- {ratio_1_2:.1f}x above {sec_name} ({sec_val:.3f})")
      else:
          top3 = ', '.join(f'{n} ({v:.3f})' for n, v in ranked[:3])
          print(f"- NEAR-TIE: {top3} -- top {min(3,len(ranked))} within {abs(best_val/ranked[min(2,len(ranked)-1)][1]):.0f}x")
      print(f"- REMOVE {worst_name}: coef={worst_val:.3f} is {abs(best_val/worst_val):.0f}x below {best_name} -- may not justify cost")
"""


def build_probe_docstring(extra_instructions: str = "") -> str:
    """Build the full probe docstring with optional project-specific extras.

    If no extra_instructions are provided, DEFAULT_EXTRA_PROBE_INSTRUCTIONS
    is used. Pass an empty string explicitly to get only the generic probe
    instructions with no extras.

    Pydantic-ai parses Args:/Returns: from docstrings and strips everything
    after them. To ensure extra_instructions are visible to the LLM, they
    are inserted BEFORE the Args: block.
    """
    body = _GENERIC_PROBE_INSTRUCTIONS_BODY
    extras = extra_instructions if extra_instructions else DEFAULT_EXTRA_PROBE_INSTRUCTIONS
    if extras:
        body += "\n\n" + extras
    return body + "\n" + _GENERIC_PROBE_ARGS
