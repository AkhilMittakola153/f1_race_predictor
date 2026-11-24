import json, os, time
from datetime import date
import requests
import pandas as pd

BASE = "https://api.jolpi.ca/ergast/f1"
CACHE = "jolpica_cache"

def get_json(path):
    os.makedirs(CACHE, exist_ok=True)
    fname = os.path.join(CACHE, path.replace("/", "_") + ".json")
    if os.path.exists(fname):
        with open(fname) as f:
            return json.load(f)

    for attempt in range(30):
        resp = requests.get(f"{BASE}/{path}.json", timeout=30)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 60))
            print(f"  rate limited, waiting {wait}s (attempt {attempt + 1})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        with open(fname, "w") as f:
            json.dump(data, f)
        time.sleep(1)
        return data

    raise RuntimeError(f"Gave up on {path} after 30 attempts")
    
def lap_to_seconds(t):
    if not t:
        return None
    try:
        parts = t.split(":")
        if len(parts) == 2:              
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])         
    except ValueError:
        return None                     
        
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
            "points": float(r["points"]),
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

def collect_season(year):
    races = get_json(str(year))["MRData"]["RaceTable"]["Races"]
    today = date.today().isoformat()
    frames = []
    for race in races:
        if race["date"] >= today:
            continue  # not raced yet
        rnd = int(race["round"])
        print(f"{year} R{rnd:02d} {race['raceName']}")
        df = race_results(year, rnd).merge(
            quali_results(year, rnd), on="driver", how="left")
        df.insert(0, "year", year)
        df.insert(1, "round", rnd)
        df.insert(2, "race", race["raceName"])
        df.insert(3, "circuit", race["Circuit"]["circuitId"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    frames = []
    for year in range(2018, 2027):
        frames.append(collect_season(year))
    
    df = pd.concat(frames, ignore_index=True)
    df.to_csv("results_2018_2026.csv", index=False)
    
    print(df.shape)
    print(df.groupby("year").size())
    print("\nRows missing best_lap, by race:")
    print(df[df["best_lap"].isna()].groupby(["year", "round"]).size()
            .loc[lambda s: s >= 5])