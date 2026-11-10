#Uses Pandas to clean up raw lineup data and fix any errors present
import pandas as pd
from pathlib import Path

def clean_lineups(input_file, output_file):
    # Read in raw data
    lineups = pd.read_csv(input_file)

    # Strip data & standardize
    lineups["festival"] = lineups["festival"].str.strip()
    lineups["artist_alias"] = lineups["artist_alias"].str.strip()
    lineups["type"] = lineups["type"].str.strip().str.lower()
    lineups["year"] = lineups["year"].astype(int)

    # Normalize and remove duplicates
    rows_before = len(lineups)
    lineups = lineups.drop_duplicates(
        subset=["festival", "year", "artist_alias", "type"]
    )
    duplicates_removed = rows_before - len(lineups)

    # Sort values again
    lineups = lineups.sort_values(
        by=["festival", "year", "artist_alias"]
    )

    #  Make directory (just in case)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Save preprocessed data
    lineups.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = "data/raw/lineups.csv"
    output_file = "data/processed/lineups_clean.csv"
    clean_lineups(input_file, output_file)

