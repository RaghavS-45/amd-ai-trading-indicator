"""
clean_data.py
Filters the raw Parquet to regular session hours, drops the first N minutes
after the open, and removes rows outside 09:30-16:00 ET.

Usage:
    python src/clean_data.py
"""

from pathlib import Path

import pandas as pd
import pytz
import yaml

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())

ET = pytz.timezone("America/New_York")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Convert to ET for time-of-day filtering
    local = df.index.tz_convert(ET)

    open_h, open_m = 9, 30
    close_h, close_m = 16, 0

    in_session = (
        (local.time >= pd.Timestamp(f"{open_h:02d}:{open_m:02d}").time())
        & (local.time < pd.Timestamp(f"{close_h:02d}:{close_m:02d}").time())
        & (local.weekday < 5)
    )
    df = df[in_session].copy()

    # Re-localise after filter
    local = df.index.tz_convert(ET)
    minutes_since_open = (local.hour * 60 + local.minute) - (open_h * 60 + open_m)
    df = df[minutes_since_open >= CFG["drop_open_mins"]]

    return df


def main():
    raw_path = ROOT / CFG["raw_data_path"]
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. Run src/download_data.py first."
        )

    df = pd.read_parquet(raw_path)
    df.index = pd.to_datetime(df.index, utc=True)
    print(f"Raw rows: {len(df):,}")

    df = clean(df)
    print(f"Clean rows (session only, -{CFG['drop_open_mins']} open mins): {len(df):,}")

    out_path = ROOT / CFG["clean_data_path"]
    df.to_parquet(out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
