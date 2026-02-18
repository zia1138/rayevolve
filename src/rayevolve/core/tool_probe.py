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
- Component ranking: order by a metric (AUC, coefficient,
  separability) to identify strongest/weakest.
- Distribution diagnostics: scale, variance, skew of outputs to
  detect imbalance or outliers.
- Processing bottlenecks: hotspots in time, memory, or IO that
  limit throughput or cause timeouts.

Using probe_hook (preferred for cross-fold analysis):
- Override `probe_hook(ctx)` -- it receives a single dict `ctx`
  with all cross-fold data, preprocessed features, trained model
  objects, and meta-learner results. Called ONCE after all folds
  and meta-learner tuning. See probe_hook() docstring in the
  system message for all available ctx keys and scopes. Example:
      def probe_hook(ctx):
          Z = ctx['Z']  # stacked OOF logits, all folds
          names = ctx['base_learner_names']
          corr = np.corrcoef(Z.T)
          # ... analysis and prints here ...

NaN pitfall:
- Correlating features with a zero-variance target (e.g., a
  minority class with 0 samples in the fold) produces NaN for ALL
  features. Always check the target count before correlating
  (`if target.sum() == 0: print("WARNING: ...")`) and use
  `.dropna()` after `.corrwith()`. The last CV fold may not
  contain all minority classes.

Output formatting (extended):
- If NOT using hooks and your override runs multiple times (once
  per fold), collect values into a global list and print a single
  aggregated summary inside the LAST call (never at module level).
- Rank-order results so the most important items appear first.
- Flag instabilities: if a value changes sign across runs or has
  high variance relative to its mean, note it explicitly.
- Before writing each Summary bullet, RANK all components by the
  metric you measured (coefficient, correlation, AUC, etc.) and
  assign verdicts based on RELATIVE position, not absolute value
  alone. A positive value does NOT mean KEEP -- if it's 10x smaller
  than the best, it may warrant REMOVE or INVESTIGATE.

  Ranking rules:
  - Show the TOP 2-3 names+values in every ranking, not just #1.
  - Use DOMINANT only when #1 is >2x #2. Otherwise use NEAR-TIE
    and list the cluster (e.g., "A (0.13) = B (0.13) = C (0.11)").
  - Show TIER STRUCTURE when it exists (e.g., "top 3 at 0.11-0.13,
    bottom 2 at 0.05-0.06").
  - When reporting engineered features, COMPARE against the
    strongest natural feature in the same model -- an eng__ coef
    only has meaning relative to natural features.

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
