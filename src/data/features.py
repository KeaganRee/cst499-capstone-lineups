import pandas as pd


def create_historical_features(input_path, output_path):
    df = pd.read_csv(input_path)

    feature_rows = []

    for _, row in df.iterrows():
        festival = row["festival"]
        year = row["year"]
        artist = row["artist_alias"]

        previous_rows = df[df["year"] < year]

        artist_history = previous_rows[
            previous_rows["artist_alias"] == artist
        ]

        festival_history = artist_history[
            artist_history["festival"] == festival
        ]

        other_festival_history = artist_history[
            artist_history["festival"] != festival
        ]

        previous_festival_rows = df[
            (df["festival"] == festival) &
            (df["year"] < year)
        ]

        previous_year_rows = df[
            (df["festival"] == festival) &
            (df["year"] == year - 1)
        ]

        if len(festival_history) > 0:
            last_appearance_year = festival_history["year"].max()
            years_since_appearance = year - last_appearance_year
        else:
            years_since_appearance = -1

        feature_rows.append({
            "festival": festival,
            "year": year,
            "artist_alias": artist,

            "artist_total_prior_appearances": len(artist_history),
            "artist_festival_prior_appearances": len(festival_history),
            "artist_other_festival_prior_appearances": len(
                other_festival_history
            ),
            "years_since_festival_appearance": years_since_appearance,
            "festival_previous_year_size": len(previous_year_rows),

            # This row represents an actual lineup appearance
            "appeared": 1,
        })

    features = pd.DataFrame(feature_rows)
    features.to_csv(output_path, index=False)

    return features


if __name__ == "__main__":
    create_historical_features(
        "data/processed/lineups_clean.csv",
        "data/processed/lineup_features.csv"
    )