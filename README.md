# F1 Race Predictor

Predicts the finishing order of a Formula 1 race after qualifying, using public results data from 2018 onward.

The model starts from the starting grid and learns a **correction**: which drivers are likely to gain or lose places, based on driver form, team form, qualifying pace and reliability. It is tested with a walk-forward backtest, where each race is predicted using only the races before it. On that test it beats the "finish = grid order" baseline in every season from 2022 to 2026.

## Results

Walk-forward backtest. Each race is predicted with a model trained only on earlier races.

| Season | Spearman, grid | Spearman, model | MAE grid (places) | MAE model (places) |
|---|---|---|---|---|
| 2022 | 0.526 | **0.574** | 3.94 | **3.71** |
| 2023 | 0.586 | **0.643** | 3.76 | **3.41** |
| 2024 | 0.730 | **0.760** | 2.90 | **2.77** |
| 2025 | 0.651 | **0.670** | 3.34 | **3.27** |
| 2026 | 0.635 | **0.679** | 3.64 | **3.49** |

- **Spearman:** how well the predicted order matches the real order (1 = perfect). Higher is better.
- **MAE:** average error in finishing places. Lower is better.

The gains are modest (2–10%). That's realistic for F1, where the grid is already a strong predictor and crashes, safety cars and rain can't be predicted from pre-race data. The largest gains come in seasons where the grid was a weaker guide: the first season of a new set of rules (2022, 2026) and 2023.

## How it works

```
Jolpica API ──► collect.py ──► results_2018_2026.csv
                                      │
                                      ▼
                               features.py  (no-leakage features)
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
               model.py                            predict.py
      (walk-forward backtest)          (qualifying for one race → predicted order)
```

### Features

| Feature | Meaning |
|---|---|
| `grid` | Starting position (qualifying order when predicting) |
| `gap_to_pole_pct` | Best qualifying lap, % slower than pole (capped at 7%) |
| `drv_form5` | Driver's average finish over the last 5 races they finished |
| `drv_dnf10` | Driver's DNF rate over the last 10 races |
| `team_form5` | Team's average finish over the last 5 races, reset at each rule change (2022, 2026) |
| `drv_track_hist` | Driver's average finish at this circuit in earlier seasons |
| `races_into_era` | Number of races since the last rule change |

Every history feature uses only earlier races (`shift(1)`), so the race being predicted never leaks into its own features.

### Model

- **Target:** places gained or lost, `position − grid`. The prediction is `grid + predicted change`, ranked within the race.
- **Algorithm:** scikit-learn `HistGradientBoostingRegressor`, kept small on purpose: 50 trees, depth 3, at least 100 samples per leaf.
- **Training data:** only drivers who finished. Retirements happen almost at random relative to pace and added noise.

[docs/METHODOLOGY.md](docs/METHODOLOGY.md) has the full experiment log: what was tried, what failed and why.

## Quick start

Requires Python 3.12. Python 3.15 crashed inside pandas during testing.

Clone this repository, open its folder, then:

```bash
python -m venv venv
# Windows: .\venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 1. Collect data (once, then after each race)

```bash
python collect.py
```

This downloads race and qualifying results for 2018 to the current season. API responses are cached in `jolpica_cache/`, so later runs only fetch new races. The first run makes about 400 requests and may pause for the API rate limit; it retries automatically.

### 2. Build features

```bash
python features.py
```

### 3. Backtest (optional)

```bash
python model.py
```

### 4. Predict a race (after qualifying)

```bash
python predict.py 2026 22
```

If the race has already happened, the real result and both Spearman scores are shown next to the prediction. Example on the 2026 Las Vegas GP, predicted using only earlier races:

```
Spearman  model: 0.651   grid: 0.633
```

Before qualifying it prints `No qualifying data yet` and stops. The API can take a few hours to update after a session.

## Project structure

| File | Purpose |
|---|---|
| `collect.py` | Downloads results and qualifying from Jolpica, with caching and retries |
| `features.py` | `build_features()`: cleaning and leakage-safe feature engineering |
| `model.py` | Model comparison and walk-forward backtest |
| `predict.py` | Predicts the finishing order for one race |
| `docs/METHODOLOGY.md` | Experiment log and design decisions |

## Limitations

- Grid penalties aren't applied: the grid is the qualifying order.
- No weather, practice pace or tyre data. FastF1 timing data now needs a paid F1TV account, so only free results data is used.
- Retirements, first-lap incidents and safety cars aren't predicted.
- Each season has only about 24 races, so winner and podium hit rates are noisy. Spearman and MAE use every driver and are more reliable.

## Data source

Results and qualifying data come from the [Jolpica-F1 API](https://github.com/jolpica/jolpica-f1), the community successor to the Ergast API. This project isn't associated with Formula 1. F1, FORMULA ONE and related marks are trademarks of Formula One Licensing B.V.

## Acknowledgements

Inspired by [f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay) by Tom Shaw.
