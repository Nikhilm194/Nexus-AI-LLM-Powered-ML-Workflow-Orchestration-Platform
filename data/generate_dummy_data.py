"""
Generate dummy electricity consumption data for testing the ML pipeline.

Creates a CSV with 100 rows containing:
  - timestamp: hourly timestamps starting from 2025-01-01
  - temperature: realistic temperature values (15–40 °C)
  - humidity: realistic humidity percentages (30–90 %)
  - kilowatt_hours: electricity consumption (50–500 kWh)

5 NaN values are intentionally injected into the temperature column
to facilitate testing of the imputation block.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────
NUM_ROWS = 100
SEED = 42
NAN_COUNT = 5  # intentional missing values in temperature
OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "dummy_electricity_data.csv"

# ── Generate data ────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

timestamps = pd.date_range(start="2025-01-01", periods=NUM_ROWS, freq="h")
temperature = rng.uniform(low=15.0, high=40.0, size=NUM_ROWS).round(2)
humidity = rng.uniform(low=30.0, high=90.0, size=NUM_ROWS).round(2)
kilowatt_hours = rng.uniform(low=50.0, high=500.0, size=NUM_ROWS).round(2)

# ── Inject NaN values into temperature ───────────────────────────────
nan_indices = rng.choice(NUM_ROWS, size=NAN_COUNT, replace=False)
temperature[nan_indices] = np.nan

# ── Build DataFrame & save ───────────────────────────────────────────
df = pd.DataFrame({
    "timestamp": timestamps,
    "temperature": temperature,
    "humidity": humidity,
    "kilowatt_hours": kilowatt_hours,
})

df.to_csv(OUTPUT_FILE, index=False)

print(f"[OK] Generated {OUTPUT_FILE}")
print(f"   Rows  : {len(df)}")
print(f"   NaNs  : {df['temperature'].isna().sum()} (in temperature)")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nNaN rows:\n{df[df['temperature'].isna()]}")
