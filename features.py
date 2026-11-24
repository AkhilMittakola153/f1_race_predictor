import pandas as pd

FEATURES = ["grid", "gap_to_pole_pct", "drv_form5", "drv_dnf10",
            "team_form5", "drv_track_hist", "races_into_era"]

ALIASES = {
    "Toro Rosso": "Racing Bulls", "AlphaTauri": "Racing Bulls",
    "RB F1 Team": "Racing Bulls",
    "Force India": "Aston Martin", "Racing Point": "Aston Martin",
    "Renault": "Alpine", "Alpine F1 Team": "Alpine",
    "Alfa Romeo": "Sauber", "Kick Sauber": "Sauber",
}

def era(year):
    if year <= 2021:
        return 2018
    if year <= 2025:
        return 2022
    return 2026

def build_features(df):
    df = df.sort_values(["year", "round", "grid"]).reset_index(drop=True)

    n_cars = df.groupby(["year", "round"])["driver"].transform("count")
    df.loc[df["grid"] == 0, "grid"] = n_cars

    df["gap_to_pole_pct"] = df["gap_to_pole_pct"].clip(upper=7)

    status = df["status"].fillna("")
    df["finished"] = (status.isin(["Finished", "Lapped"])
                      | status.str.startswith("+")).astype(int)
    df["pos_if_finished"] = df["position"].where(df["finished"] == 1)

    g = df.groupby("driver")
    df["drv_form5"] = g["pos_if_finished"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["drv_dnf10"] = 1 - g["finished"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean())

    df["team_key"] = df["team"].replace(ALIASES)
    df["era"] = df["year"].map(era)

    team = (df.groupby(["era", "team_key", "year", "round"], as_index=False)
              ["pos_if_finished"].mean()
              .sort_values(["year", "round"]))
    team["team_form5"] = team.groupby(["era", "team_key"])["pos_if_finished"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df = df.merge(team[["era", "team_key", "year", "round", "team_form5"]],
                  on=["era", "team_key", "year", "round"], how="left")

    df["drv_track_hist"] = df.groupby(["driver", "circuit"])["pos_if_finished"].transform(
        lambda s: s.shift(1).expanding().mean())

    races = (df[["era", "year", "round"]].drop_duplicates()
               .sort_values(["year", "round"]))
    races["races_into_era"] = races.groupby("era").cumcount()
    df = df.merge(races, on=["era", "year", "round"], how="left")

    return df


if __name__ == "__main__":
    raw = pd.read_csv("results_2018_2026.csv")
    df = build_features(raw)
    print(df[FEATURES].describe().round(2).T)
    df.to_csv("features.csv", index=False)