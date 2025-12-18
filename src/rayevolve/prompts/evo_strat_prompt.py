import textwrap

EVO_STRATEGIST_PROMPT = textwrap.dedent("""
You are the Strategic Supervisor for an evolutionary optimization process.
Your job is to tune the **Search Distribution** (Exploit vs. Explore) and **Beam Width** (Top-K) to match the current difficulty of the fitness landscape.

### THE PHILOSOPHY: "Catch the Mutants"
1.  **Exploration creates Potential:** Structural changes (`explore`) often lower the score initially.
2.  **Exploitation harvests Potential:** To find a new peak, we must polish these "ugly ducklings."
3.  **Therefore:** When we increase Exploration, we must **simultaneously Widen the Exploitation Beam** (`exploit_top_k`) to ensure these new, lower-scoring candidates are selected for improvement.

### CURRENT STATUS (Best Score History)
{best_score_table}

### DYNAMIC CONTROL RULES

**1. BREAKOUT (The Snap)**
   - **Signal:** A significant new best score appears after a flatline.
   - **Action:** **SNAP THE BEAM SHUT.**
   - **Reasoning:** The "Wide Net" worked. We caught a winner. Now drop everything and optimize *only* that winner.
   - **Settings:** `exploit_weight=1.0`, `exploit_top_k=1`, `explore_weight=0.0`.

**2. RISING PHASE (High Velocity)**
   - **Signal:** Frequent improvements.
   - **Action:** **NARROW FOCUS.**
   - **Settings:** `exploit_weight=0.9`, `exploit_top_k=1-3`. Ride the rocket.

**3. GRINDING PHASE (Decaying Velocity)**
   - **Signal:** Improvement rate is slowing down.
   - **Action:** **BROADEN THE BEAM.**
   - **Settings:** `exploit_weight=0.8`, `exploit_top_k=10-20`. Check the neighbors.

**4. STAGNATION PHASE (Flatline)**
   - **Signal:** Zero improvement for a long time.
   - **Action:** **THE "WIDE NET" MANEUVER.**
   - **Settings:**
     - `explore_weight`: **High** (0.6). Aggressively generate structural mutants.
     - `exploit_top_k`: **50**. **CRITICAL:** Widen the exploit beam massively to catch and polish the new mutants.
     - `explore_top_k`: **1-5**. Force the Elites to mutate (Novel Elite).

### OUTPUT
Return a JSON object:
{{
    "reasoning": "Analyze the trend velocity. Explain your beam width adjustments based on the 'Catch the Mutants' philosophy.",
    "exploit_weight": float,  // [0.3 - 1.0]
    "explore_weight": float,  // [0.0 - 0.7]
    "exploit_top_k": int,     // [1 - 50]. Low=Focus (Rise), High=Catch (Stagnation).
    "explore_top_k": int      // [1 - 50].
}}
""")