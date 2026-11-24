# Methodology and experiment log

This document records how the model was built: each experiment, its result, and what it showed. Experiments were changed **one variable at a time**, and each was compared against the same baseline.

## 1. Problem setup

**Goal:** predict a race's finishing order once qualifying is known.

**Baseline:** "finish = grid order". Over 2025 this baseline already gives:

| Metric | 2025 |
|---|---|
| Spearman (grid vs finish) | 0.652 |
| Pole sitter wins | 67% of races |

Any model has to beat this to be useful.

**Metrics** (averaged over the races in a season):

| Metric | Meaning |
|---|---|
| `spearman` | Rank correlation between predicted and real order. Higher is better |
| `mae` | Mean absolute error in places. Lower is better |
| `winner` | Share of races where the predicted winner won |
| `podium` | Share of the real podium that was predicted |

With about 24 races a season, one race moves `winner` by about 4%. `spearman` and `mae` use all ~20 drivers per race and are the main metrics.

## 2. Data

**Source:** the Jolpica-F1 API (results and qualifying, 2018 onward). The plan was to use FastF1, but in October 2026 Formula 1 put its timing data behind an F1TV login, so only free results data is used.

**Collection issues found and handled:**

| Issue | Handling |
|---|---|
| 2020 Sakhir GP lap times under 1 minute (`"53.377"`, no minutes) | `lap_to_seconds` handles both formats; anything else becomes missing |
| HTTP 429 rate limiting | Retry loop that honours `Retry-After`, plus local caching |
| Missing qualifying times (e.g. 2025 Miami) | Left as NaN; the model handles missing values |
| Wet qualifying outliers (2020 Turkey: 31% off pole; 2023 Saudi: 45%) | Gap to pole capped at 7%, in line with F1's 107% rule |
| Pit-lane starts recorded as grid 0 | Replaced with last place |
| Team renames (Toro Rosso → AlphaTauri → RB, etc.) | Mapped to one team key |

**Final dataset:** 3,920 driver-race rows, 2018–2026 (2026: 22 cars per race).

## 3. Leakage control

Every history feature is computed with `shift(1)` before any rolling average, so a row's features use only earlier races. This is the same idea as a causal filter: the output at sample *n* depends only on samples before *n*.

Team form also **resets at each rule change** (2022, 2026), because the pecking order changes with the cars.

## 4. Experiment log

### E1: First model

Gradient boosting (300 trees) predicting finishing position. Trained on 2018–2025, tested on 2026.

| | Spearman | MAE |
|---|---|---|
| Grid | 0.635 | 3.64 |
| Model | 0.608 | 4.13 |

**The model lost to the baseline.** Hypothesis: the 2026 rule change makes older patterns misleading.

### E2: Test in a stable era

Same model, tested on 2024 and 2025 (no rule change).

| Season | Grid MAE | Model MAE |
|---|---|---|
| 2024 | 2.90 | 3.45 |
| 2025 | 3.34 | 3.77 |

It still lost, so **the hypothesis was rejected**. The rule change wasn't the main cause.

### E3: Ablation

Features added one group at a time. The "grid only" model matched the baseline, and **every added feature group made it worse**. That pointed to a problem in the training setup rather than in any one feature.

### E4: Train only on finishers

Retirements put fast drivers at P15–P20 almost at random, and the model was fitting that noise.

| Season | MAE, all rows | MAE, finishers only |
|---|---|---|
| 2024 | 3.45 | 3.13 (−9%) |
| 2025 | 3.77 | 3.45 (−8%) |

**A clear improvement.**

### E5: DNF-free form features

Driver, team and track form recalculated from finished races only. Reliability stays covered by `drv_dnf10`.

Result: a small change (2024 MAE 3.13 → 3.03; 2025 3.45 → 3.49), within noise.

### E6: Residual target

Predict `position − grid` instead of `position`, and add the prediction back onto the grid.

Result: no meaningful change.

### E7: Model complexity

Same features and residual target, three model sizes:

| Model | 2024 Spearman | 2024 MAE | 2025 Spearman | 2025 MAE |
|---|---|---|---|---|
| Grid baseline | 0.730 | 2.90 | 0.651 | 3.34 |
| Boosting, 300 trees | 0.721 | 3.06 | 0.642 | 3.46 |
| **Boosting, 50 trees, depth 3** | **0.757** | **2.80** | **0.666** | **3.30** |
| Ridge (linear) | 0.747 | 2.86 | 0.655 | 3.37 |

**Overfitting confirmed:** the simpler models generalise better. The small booster was chosen.

**Ridge coefficients** (effect on places gained or lost, per standard deviation of each feature):

| Feature | Coefficient | Reading |
|---|---|---|
| `grid` | −3.74 | Cars further back gain places; front-runners can only lose them |
| `team_form5` | +1.12 | Weaker teams lose places compared with their grid slot |
| `drv_form5` | +1.10 | Same for drivers in poor form |
| `gap_to_pole_pct` | +0.24 | Small effect |
| `drv_track_hist` | +0.14 | Small effect |
| `races_into_era` | +0.11 | Small effect |
| `drv_dnf10` | +0.03 | About zero, as expected when training only on finishers |

In plain words: a fast driver or team that qualified badly tends to recover places in the race.

### E8: Walk-forward backtest (final evaluation)

For every race from 2022 to 2026, a new model is trained on all earlier races (including earlier races in the same season) and then predicts that race.

| Season | Spearman grid → model | MAE grid → model |
|---|---|---|
| 2022 | 0.526 → 0.574 (+9.1%) | 3.94 → 3.71 (−6.0%) |
| 2023 | 0.586 → 0.643 (+9.7%) | 3.76 → 3.41 (−9.3%) |
| 2024 | 0.730 → 0.760 (+4.1%) | 2.90 → 2.77 (−4.3%) |
| 2025 | 0.651 → 0.670 (+2.9%) | 3.34 → 3.27 (−2.1%) |
| 2026 | 0.635 → 0.679 (+6.9%) | 3.64 → 3.49 (−4.2%) |

**The model beats the grid in all five seasons.** Walk-forward training also fixed the problem from E1: with earlier 2026 races in training, 2026 Spearman rose from 0.608 to 0.679.

### E9: Out-of-sample example

2026 Las Vegas GP, predicted using only earlier races:

| | Spearman |
|---|---|
| Grid | 0.633 |
| Model | 0.651 |

The model correctly moved both Ferraris up (from P12 to P8 and from P11 to P9; they finished P7 and P10) and ranked Norris ahead of Russell (they finished P2 and P9). It didn't foresee the incidents that dropped two top-6 starters to P15 and P20.

## 5. Lessons

| Lesson | Where it came from |
|---|---|
| Always compare against a simple baseline | E1: a "working" model was worse than the grid |
| Test hypotheses, don't assume them | E2 rejected the rule-change explanation |
| Change one thing at a time | E3–E7 |
| Noise in the target can outweigh signal in the features | E4 |
| Simpler models often generalise better | E7 |
| Evaluate the way the model will be used | E8 (walk-forward) |

## 6. Possible next steps

| Idea | Expected benefit |
|---|---|
| Sprint race result as a feature | A race-pace signal on sprint weekends |
| Final grid including penalties | Fixes cases where grid ≠ qualifying order |
| Wet-race flag | Rain reshuffles results |
| A ranking model (LightGBM LambdaRank) | Optimises the order directly |
