import pandas as pd
from pathlib import Path


def clean_lineups(input_file, output_file):
    lineups = pd.read_csv(input_file)

    lineups["festival"] = lineups["festival"].str.strip()
    lineups["artist_alias"] = lineups["artist_alias"].str.strip()
    lineups["type"] = lineups["type"].str.strip().str.lower()
    lineups["year"] = lineups["year"].astype(int)

    expanded_rows = []

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

    cleaned_lineups = pd.DataFrame(expanded_rows)

    # Prevent an artist from appearing more than once at the same festival
    cleaned_lineups = cleaned_lineups.drop_duplicates(
        subset=["festival", "year", "artist_alias"]
    )

    cleaned_lineups = cleaned_lineups.sort_values(
        by=["festival", "year", "artist_alias"]
    )

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    cleaned_lineups.to_csv(output_file, index=False)


if __name__ == "__main__":
    clean_lineups(
        "data/raw/lineups.csv",
        "data/processed/lineups_clean.csv"
    )