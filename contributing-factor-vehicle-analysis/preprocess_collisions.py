import pandas as pd
import re
from pathlib import Path

# ===================== CONFIG =====================

# Path to your original NYC collisions CSV
INPUT_CSV = Path("/Volumes/T7/MAC/COLLEGE/vizualization/NYC_Collisions_Project/data/Motor_Vehicle_Collisions_-_Crashes_20251206.csv")  # <-- change if needed

# Path to the cleaned CSV you'll load into Tableau
OUTPUT_CSV = Path("Motor_Vehicle_Collisions_clean_for_tableau.csv")

# How many vehicle types & factors you want to keep as named categories
TOP_VEHICLE_TYPES = 12
TOP_FACTORS = 20

# ==================================================


def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} rows")
    return df


# ---------- Helpers for cleaning text ----------

def normalize_text(s: str) -> str:
    """
    Lowercase, strip, collapse whitespace.
    Return None for empty or NaN-like values.
    """
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    if s in ("", "nan", "null", "none"):
        return None
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s


# ---------- Vehicle type cleaning ----------

def clean_vehicle_type(raw: str) -> str | None:
    """
    Map messy vehicle type strings into a smaller set of categories.
    Returns a cleaned vehicle type label or None.
    """
    s = normalize_text(raw)
    if s is None:
        return None

    # remove generic junk
    if s in {"unknown", "unk", "unspecified", "other", "misc"}:
        return None

    # simple pattern-based grouping
    if any(k in s for k in ["taxi", "cab", "yellow cab"]):
        return "taxi"

    if any(k in s for k in ["van", "minivan"]):
        return "van"

    if any(k in s for k in ["bus", "school bus"]):
        return "bus"

    if any(k in s for k in ["motorcycle", "motor bike", "motorbike", "mcycle", "scooter"]):
        return "motorcycle"

    if any(k in s for k in ["pickup", "pick-up", "pick up"]):
        return "pickup truck"

    if any(k in s for k in ["suv", "sport utility", "sp util", "sport util", "jeep"]):
        return "suv"

    if any(k in s for k in ["sedan", "4 dr", "4dr", "4 door", "2 dr", "2dr", "sdn", "sdan"]):
        return "sedan"

    if any(k in s for k in ["box truck", "boxtruck", "straight truck"]):
        return "box truck"

    if "bike" in s or "bicycle" in s:
        return "bike"

    if "passeng" in s:
        return "passenger vehicle"

    # if nothing matched, return the normalized string as-is
    return s


def add_clean_vehicle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create vehicle_type_clean and vehicle_type_final columns based on Vehicle Type Code 1.
    """
    # Try a few possible column name variants
    vehicle_col_candidates = [
        "Vehicle Type Code 1",
        "VEHICLE TYPE CODE 1",
        "vehicle_type_code_1",
    ]
    vehicle_col = next((c for c in vehicle_col_candidates if c in df.columns), None)
    if vehicle_col is None:
        raise KeyError("Could not find Vehicle Type Code 1 column. Check your CSV headers.")

    print(f"Cleaning vehicle types from column: {vehicle_col}")

    df["vehicle_type_clean"] = df[vehicle_col].map(clean_vehicle_type)

    # Drop rows where vehicle type is missing after cleaning
    before = len(df)
    df = df[df["vehicle_type_clean"].notna()].copy()
    print(f"Dropped {before - len(df):,} rows with unknown/empty vehicle type")

    # Keep only top N vehicle types; group the rest as "other"
    top_types = df["vehicle_type_clean"].value_counts().head(TOP_VEHICLE_TYPES).index
    print("Top vehicle types:", list(top_types))

    df["vehicle_type_final"] = df["vehicle_type_clean"].where(
        df["vehicle_type_clean"].isin(top_types),
        other="other"
    )

    return df


# ---------- Contributing factor cleaning ----------

def clean_factor(raw: str) -> str | None:
    """
    Clean primary contributing factor text.
    """
    s = normalize_text(raw)
    if s is None:
        return None

    if s in {"unspecified", "unknown", "unk"}:
        return None

    # unify some common variants
    s = s.replace(" driver inattention/distraction", "driver inattention/distraction")

    # you can add more specific mappings here if you want

    return s


def add_clean_factor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create factor_clean and factor_final from Contributing Factor Vehicle 1.
    """
    factor_col_candidates = [
        "Contributing Factor Vehicle 1",
        "CONTRIBUTING FACTOR VEHICLE 1",
        "contributing_factor_vehicle_1",
    ]
    factor_col = next((c for c in factor_col_candidates if c in df.columns), None)
    if factor_col is None:
        raise KeyError("Could not find Contributing Factor Vehicle 1 column. Check your CSV headers.")

    print(f"Cleaning contributing factors from column: {factor_col}")

    df["factor_clean"] = df[factor_col].map(clean_factor)

    # Drop rows where factor is missing after cleaning
    before = len(df)
    df = df[df["factor_clean"].notna()].copy()
    print(f"Dropped {before - len(df):,} rows with unknown/empty factor")

    # Keep only top N factors; group the rest as "other / rare"
    top_factors = df["factor_clean"].value_counts().head(TOP_FACTORS).index
    print("Top contributing factors:", list(top_factors))

    df["factor_final"] = df["factor_clean"].where(
        df["factor_clean"].isin(top_factors),
        other="other / rare"
    )

    return df


# ---------- Severity category ----------

def add_severity_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a categorical severity column based on injury and fatality counts.
    """
    # Try several possible column name variants
    injured_candidates = [
        "Number Of Persons Injured",
        "NUMBER OF PERSONS INJURED",
        "NUMBER OF PERSONS INJURED ",
    ]
    killed_candidates = [
        "Number Of Persons Killed",
        "NUMBER OF PERSONS KILLED",
        "NUMBER OF PERSONS KILLED ",
    ]

    injured_col = next((c for c in injured_candidates if c in df.columns), None)
    killed_col = next((c for c in killed_candidates if c in df.columns), None)

    if injured_col is None or killed_col is None:
        raise KeyError("Could not find 'Number Of Persons Injured/Killed' columns. Check your CSV headers.")

    print(f"Using injury column: {injured_col}")
    print(f"Using fatality column: {killed_col}")

    # Ensure numeric
    df[injured_col] = pd.to_numeric(df[injured_col], errors="coerce").fillna(0).astype(int)
    df[killed_col] = pd.to_numeric(df[killed_col], errors="coerce").fillna(0).astype(int)

    def categorize(row):
        if row[killed_col] > 0:
            return "Fatal"
        elif row[injured_col] > 0:
            return "Injury"
        else:
            return "Property Damage Only"

    df["severity_category"] = df.apply(categorize, axis=1)
    return df


# ---------- Optional: derive time fields (if teammates want to use later) ----------

def add_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    (Optional) Add crash_date, crash_hour for time-based analysis.
    Won't break anything if some columns are missing; it's just a bonus.
    """
    date_candidates = ["Crash Date", "CRASH DATE"]
    time_candidates = ["Crash Time", "CRASH TIME"]

    date_col = next((c for c in date_candidates if c in df.columns), None)
    time_col = next((c for c in time_candidates if c in df.columns), None)

    if date_col:
        df["crash_date"] = pd.to_datetime(df[date_col], errors="coerce")

    if time_col:
        # assume HH:MM format
        t = pd.to_datetime(df[time_col], format="%H:%M", errors="coerce")
        df["crash_hour"] = t.dt.hour

    return df


# ---------- Reduce columns for Tableau ----------

def keep_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns that are actually useful for Tableau visualizations.
    You can add/remove columns here as your team needs.
    """
    keep = []

    for col in [
        "Collision Id", "COLLISION_ID",
        "Crash Date", "CRASH DATE",
        "Crash Time", "CRASH TIME",
        "Borough", "BOROUGH",
        "Zip Code", "ZIP CODE",
        "Latitude", "LATITUDE",
        "Longitude", "LONGITUDE",
        "Number Of Persons Injured", "NUMBER OF PERSONS INJURED",
        "Number Of Persons Killed", "NUMBER OF PERSONS KILLED",
        "Vehicle Type Code 1", "VEHICLE TYPE CODE 1",
        "Contributing Factor Vehicle 1", "CONTRIBUTING FACTOR VEHICLE 1",
    ]:
        if col in df.columns:
            keep.append(col)

    # derived columns we created
    keep += [
        "vehicle_type_clean",
        "vehicle_type_final",
        "factor_clean",
        "factor_final",
        "severity_category",
    ]

    if "crash_date" in df.columns:
        keep.append("crash_date")
    if "crash_hour" in df.columns:
        keep.append("crash_hour")

    # Drop duplicates in keep list
    keep = list(dict.fromkeys(keep))

    print("Keeping columns:", keep)
    return df[keep].copy()


def main():
    df = load_data(INPUT_CSV)

    df = add_severity_column(df)
    df = add_clean_vehicle_columns(df)
    df = add_clean_factor_columns(df)
    df = add_time_fields(df)           # optional but useful for the team
    df = keep_relevant_columns(df)

    print(f"Final cleaned dataset: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Saving cleaned data to {OUTPUT_CSV} ...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
