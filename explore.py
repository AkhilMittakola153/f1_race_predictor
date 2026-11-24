import requests
import pandas as pd

BASE = "https://api.jolpi.ca/ergast/f1"

def get_json(path):
    return requests.get(f"{BASE}/{path}.json", timeout=30).json()

def lap_to_seconds(t):
    if not t:
        return None
    m, s = t.split(":")
    return int(m) * 60 + float(s)

def race_results(year, rnd):
    race = get_json(f"{year}/{rnd}/results")["MRData"]["RaceTable"]["Races"][0]
    rows = []
    for r in race["Results"]:
        rows.append({
            "driver": r["Driver"]["code"],
            "team": r["Constructor"]["name"],
            "grid": int(r["grid"]),
            "position": int(r["position"]),
            "status": r["status"],
        })
    return pd.DataFrame(rows)

def quali_results(year, rnd):
    race = get_json(f"{year}/{rnd}/qualifying")["MRData"]["RaceTable"]["Races"][0]
    rows = []
    for q in race["QualifyingResults"]:
        times = [lap_to_seconds(q.get(k)) for k in ("Q1", "Q2", "Q3")]
        times = [t for t in times if t]
        rows.append({
            "driver": q["Driver"]["code"],
            "quali_pos": int(q["position"]),
            "best_lap": min(times) if times else None,
        })
    df = pd.DataFrame(rows)
    df["gap_to_pole_pct"] = (df["best_lap"] / df["best_lap"].min() - 1) * 100
    return df

res = race_results(2025, 1)
quali = quali_results(2025, 1)
df = res.merge(quali, on="driver", how="left")
print(df.round(3).to_string(index=False))