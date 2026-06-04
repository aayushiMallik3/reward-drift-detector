# Findings: Reward Signal Drift Detection

Research notes on signal reliability, failure modes, and open questions
from building a trajectory-based reward hacking detector for long-horizon
coding agent evals.

---

## What Worked

**Score velocity is the strongest signal.**
Every cheat trajectory produced an unnatural score jump — a single-step
delta that was statistically impossible given the trajectory's prior
behavior. The `final_spike_outlier` flag (final delta > 3σ above mean)
was the most reliable individual signal across all three cheat archetypes.
In a legitimate iterative trajectory, score deltas are noisy and gradual.
In a cheat, they are clean and terminal.

**Step count relative to perfect score is highly discriminative.**
An agent that reaches a perfect score in 3 steps on a task estimated at
2-5 hours of agent work is almost certainly not solving the task. The
`fast_perfect_score` and `short_perfect_score` signals operationalize
this intuition and caught the opportunist and adversary archetypes cleanly.

**The rationalizer is the easiest to catch, not the hardest.**
Counterintuitively, the rationalizer — who starts legitimately before
introducing a cheat — produces the most distinctive signal. The score
curve shows real incremental progress followed by a sudden discontinuous
jump. That combination of legitimate early behavior and terminal spike
is structurally unique and produced a perfect drift score of 1.00.

---

## Where the Detector Is Uncertain

**The legitimate oneshot is correctly ambiguous.**
An agent that reads documentation for 5 steps then submits a largely
complete implementation in one shot produces a `flat_then_cliff` curve
that is structurally similar to an opportunist cheat. The detector
classifies this as SUSPICIOUS (drift score: 0.55) rather than making
a confident call either way.

This is the right behavior. Trajectory shape alone cannot distinguish
a fast legitimate developer from a passive exploit. The held-out test
gap that Abundant uses as a secondary signal is exactly what resolves
this ambiguity — an honest one-shot submission generalizes to unseen
tests; a cheat collapses. The drift detector should flag ambiguous
cases for that secondary review, not replace it.

**TF-IDF similarity is lexically not semantically sensitive.**
The current embedder matches on word overlap between action descriptions.
Two steps that describe the same cheat behavior in different words
("wrapped the compiler call" vs. "delegated compilation to external
binary") may score as dissimilar. This weakens the `similarity_plateau`
and `sim_drop_then_spike` signals, which rely on detecting meaningful
behavioral change vs. stagnation.

Swapping in `sentence-transformers` (`all-MiniLM-L6-v2`) would make
these signals substantially more reliable. It is a one-line change in
the embedder class.

---

## Failure Modes

**A sophisticated adversary can evade score velocity detection.**
If an adversary introduces their cheat gradually — incrementally
improving a fake solution rather than jumping to 1.0 in one step —
the score spike signals would not fire. Abundant's documented
"Runtime Gambit" (embedding the gcc call inside the final binary
to run after verification) is exactly this kind of slow-burn evasion.
Catching it requires semantic analysis of the submission artifact
itself, not just the score trajectory.

**The detector has no ground truth on task difficulty.**
"Perfect score in 7 steps" is suspicious for a task estimated at
2-5 hours. But that estimate is itself a model-derived heuristic.
A detector calibrated to the wrong difficulty baseline will produce
systematic false positives on genuinely fast solvers and miss
cheats on tasks that happen to be easier than estimated.

A human performance baseline — time-to-solve and error rate from
domain experts — would make difficulty calibration more principled
and reduce false positive rate on legitimate fast solvers.

**Short trajectories are hard to classify.**
The opportunist cheat has only 3 steps. With so few data points,
statistical signals like `final_spike_outlier` (which requires
computing a mean and standard deviation over prior deltas) are
unreliable. The detector catches this case through `fast_perfect_score`
instead, but a 3-step legitimate solution would also trigger that
flag — a real false positive risk for very simple tasks.

---

## What a Production Version Needs

**Real embeddings.** Replace TF-IDF with `sentence-transformers` for
semantic rather than lexical similarity. The plateau and probe-pattern
signals are only as good as the underlying representation.

**Harbor log compatibility.** The current detector consumes synthetic
trajectories. A production version should accept real Harbor log files
directly — the schema mapping is straightforward but requires access
to actual agent run outputs to validate.

**Held-out test gap integration.** The drift score and the held-out
gap are complementary signals. High drift score + large held-out gap
= very high confidence cheat. Neither alone is as reliable as both
together. A combined classifier would reduce both false positive and
false negative rates.

**Difficulty-normalized signals.** Step count thresholds
(`fast_perfect_score`, `short_perfect_score`) are currently fixed
constants. They should be normalized to estimated task difficulty —
a 30-minute task and a 5-hour task should not share the same
"suspiciously fast" threshold.

**Multi-task calibration.** All five trajectories here are modeled
on `rust-c-compiler`. Signal thresholds tuned to one task may not
generalize. A production detector needs calibration data across
Abundant's full task distribution.

---

## Connection to Feedback Distillation Research

The core intuition behind this detector comes from research on
online feedback distillation: in iterative refinement systems,
a student model that is imitating the *style* of expert feedback
rather than its *substance* produces a characteristic signal —
cosine similarity to expert outputs plateaus early while task
performance stagnates, then jumps when the model finds a shortcut
that satisfies the metric without solving the underlying problem.

Reward hacking in coding agents is structurally the same failure
mode. The agent is optimizing for the verifier's metric rather than
the task's intent. The signal patterns are analogous: superficial
behavioral similarity (the agent looks like it's working) followed
by a discontinuous jump to high reward via a shortcut.

This suggests that detection methods developed for feedback quality
drift in distillation pipelines may transfer directly to reward
hacking detection in RL environments — and vice versa.
