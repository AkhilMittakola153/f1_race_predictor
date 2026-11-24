import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("features.csv")

FEATURES = ["grid", "gap_to_pole_pct", "drv_form5", "drv_dnf10",
            "team_form5", "drv_track_hist", "races_into_era"]

def make_model(kind):
    if kind == "boost (big)":
        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, min_samples_leaf=30, random_state=42)
    if kind == "boost (small)":
        return HistGradientBoostingRegressor(
            max_iter=50, learning_rate=0.05, max_depth=3,
            min_samples_leaf=100, random_state=42)
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler(), Ridge(alpha=10))

def evaluate(data, col):
    rows = []
    for rnd, race in data.groupby("round"):
        pred_rank = race[col].rank(method="first")
        pred_pod = set(race.nsmallest(3, col)["driver"])
        real_pod = set(race.nsmallest(3, "position")["driver"])
        rows.append({
            "spearman": pred_rank.corr(race["position"], method="spearman"),
            "winner": float(race.loc[race[col].idxmin(), "position"] == 1),
            "podium": len(pred_pod & real_pod) / 3,
            "mae": (pred_rank - race["position"]).abs().mean(),
        })
    return pd.DataFrame(rows).mean()

def run(test_year, kind):
    train = df[(df["year"] < test_year) & (df["finished"] == 1)]
    test = df[df["year"] == test_year].copy()
    model = make_model(kind)
    model.fit(train[FEATURES], train["position"] - train["grid"])
    test["score"] = test["grid"] + model.predict(test[FEATURES])
    test["pred_pos"] = test.groupby("round")["score"].rank(method="first")
    return evaluate(test, "pred_pos")

def walk_forward(year, kind="boost (small)"):
    season = df[df["year"] == year]
    preds = []
    for rnd in sorted(season["round"].unique()):
        past = (df["year"] < year) | ((df["year"] == year) & (df["round"] < rnd))
        train = df[past & (df["finished"] == 1)]
        race = season[season["round"] == rnd].copy()
        model = make_model(kind)
        model.fit(train[FEATURES], train["position"] - train["grid"])
        race["pred_pos"] = (race["grid"]
                            + model.predict(race[FEATURES])).rank(method="first")
        preds.append(race)
    return evaluate(pd.concat(preds), "pred_pos")

rows = []
for year in range(2022, 2027):
    g = evaluate(df[df["year"] == year], "grid")
    m = walk_forward(year)
    rows.append({"year": year,
                 "grid_spear": g["spearman"], "model_spear": m["spearman"],
                 "grid_mae": g["mae"], "model_mae": m["mae"]})

print(pd.DataFrame(rows).round(3).to_string(index=False))