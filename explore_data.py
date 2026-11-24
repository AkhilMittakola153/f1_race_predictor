import pandas as pd

df = pd.read_csv("season_2025.csv")

print("Races without 20 drivers:")
print(df.groupby("round").size().loc[lambda s: s != 20])

print("\nMissing quali data:")
print(df[df["quali_pos"].isna()][["round", "race", "driver", "grid"]])

print("\nMissing values per column:")
print(df.isna().sum())

print("\nStatus counts:")
print(df["status"].value_counts())

print("\nPit-lane starts (grid 0):")
print(df[df["grid"] == 0][["round", "race", "driver", "position"]])

rho = df["grid"].corr(df["position"], method="spearman")
pole_win = (df.loc[df["grid"] == 1, "position"] == 1).mean()
print(f"\nGrid vs finish correlation: {rho:.3f}")
print(f"Pole sitter won: {pole_win:.0%} of races")