import os
import sys
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from collect import get_json, lap_to_seconds, CACHE
from features import build_features, FEATURES

YEAR = int(sys.argv[1])
ROUND = int(sys.argv[2])

# ---- 1. qualifying for the race we want to predict ----
path = f"{YEAR}/{ROUND}/qualifying"
races = get_json(path)["MRData"]["RaceTable"]["Races"]
if not races:
    os.remove(os.path.join(CACHE, path.replace("/", "_") + ".json"))
    sys.exit(f"No qualifying data yet for {YEAR} round {ROUND}. Try after qualifying.")
race = races[0]

rows = []
for q in race["QualifyingResults"]:
    times = [lap_to_seconds(q.get(k)) for k in ("Q1", "Q2", "Q3")]
    times = [t for t in times if t]
    rows.append({
        "year": YEAR, "round": ROUND,
        "race": race["raceName"], "circuit": race["Circuit"]["circuitId"],
        "driver": q["Driver"]["code"], "team": q["Constructor"]["name"],
        "grid": int(q["position"]),
        "position": float("nan"), "status": "", "points": float("nan"),
        "quali_pos": int(q["position"]),
        "best_lap": min(times) if times else None,
    })
upcoming = pd.DataFrame(rows)
upcoming["gap_to_pole_pct"] = (upcoming["best_lap"] / upcoming["best_lap"].min() - 1) * 100

# ---- 2. history: everything before this race ----
hist = pd.read_csv("results_2018_2026.csv")
actual = hist[(hist["year"] == YEAR) & (hist["round"] == ROUND)]
before = (hist["year"] < YEAR) | ((hist["year"] == YEAR) & (hist["round"] < ROUND))
hist = hist[before]

# ---- 3. features for history + this race together ----
df = build_features(pd.concat([hist, upcoming], ignore_index=True))
train = df[(df["year"] != YEAR) | (df["round"] != ROUND)]
train = train[train["finished"] == 1]
target = df[(df["year"] == YEAR) & (df["round"] == ROUND)].copy()

# ---- 4. train the small model and predict ----
model = HistGradientBoostingRegressor(
    max_iter=50, learning_rate=0.05, max_depth=3,
    min_samples_leaf=100, random_state=42)
model.fit(train[FEATURES], train["position"] - train["grid"])

target["score"] = target["grid"] + model.predict(target[FEATURES])
target["pred_pos"] = target["score"].rank(method="first").astype(int)
target = target.sort_values("pred_pos")

# ---- 5. show the prediction (and the real result, if it exists) ----
out = target[["pred_pos", "driver", "team", "grid"]]
if not actual.empty:
    out = out.merge(actual[["driver", "position"]], on="driver", how="left")
    out = out.rename(columns={"position": "actual"})

print(f"\n{race['raceName']} {YEAR} - predicted finishing order\n")
print(out.to_string(index=False))

if not actual.empty:
    rho = out["pred_pos"].corr(out["actual"], method="spearman")
    rho_grid = out["grid"].corr(out["actual"], method="spearman")
    print(f"\nSpearman  model: {rho:.3f}   grid: {rho_grid:.3f}")

out.to_csv(f"prediction_{YEAR}_r{ROUND:02d}.csv", index=False)