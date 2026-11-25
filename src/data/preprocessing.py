"""Data Preprocessing

Cleans the data before feeding it into the model for the feature engineering."""

import pandas as pd
from pathlib import Path


def clean_lineups(input_file, output_file):
    lineups = pd.read_csv(input_file)
    #Remove white spaces and change year to int value
    lineups["festival"] = lineups["festival"].str.strip()
    lineups["artist_alias"] = lineups["artist_alias"].str.strip()
    lineups["type"] = lineups["type"].str.strip().str.lower()
    lineups["year"] = lineups["year"].astype(int)

    expanded_rows = []
    #Separate back to back artists so that they are their own row
    for _, row in lineups.iterrows():
        if row["type"] == "b2b":
            artists = row["artist_alias"].split(" B2B ")

            for artist in artists:
                expanded_rows.append({
                    "festival": row["festival"],
                    "year": row["year"],
                    "artist_alias": artist.strip()
                })
        else:
            expanded_rows.append({
                "festival": row["festival"],
                "year": row["year"],
                "artist_alias": row["artist_alias"]
            })
    #Save edited rows into a new dataframe
    cleaned_lineups = pd.DataFrame(expanded_rows)
    #Remove duplicate values and sort dataframe on festival name then year then alias
    cleaned_lineups = cleaned_lineups.drop_duplicates(
        subset=["festival", "year", "artist_alias"]
    )
    cleaned_lineups = cleaned_lineups.sort_values(
        by=["festival", "year", "artist_alias"]
    )
    #Save into csv file in data folder
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    cleaned_lineups.to_csv(output_file, index=False)


if __name__ == "__main__":
    clean_lineups(
        "data/raw/lineups.csv",
        "data/processed/lineups_clean.csv"
    )