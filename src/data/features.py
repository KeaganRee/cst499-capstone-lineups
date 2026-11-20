import pandas as pd


def create_historical_features(input_path, output_path):
    df = pd.read_csv(input_path)

    feature_rows = []

    festivals = df["festival"].unique()
    years = sorted(df["year"].unique())

    for festival in festivals:
        for year in years:
            # Only use information available before the target year
            previous_rows = df[df["year"] < year]

            # Artists known before this year
            previous_artists = set(
                previous_rows["artist_alias"]
            )

            # Artists who actually appeared this year
            current_artists = set(
                df[
                    (df["festival"] == festival) &
                    (df["year"] == year)
                ]["artist_alias"]
            )

            # Include both positive and negative candidates
            candidate_artists = previous_artists.union(current_artists)

            previous_year_rows = df[
                (df["festival"] == festival) &
                (df["year"] == year - 1)
            ]

            for artist in candidate_artists:
                artist_history = previous_rows[
                    previous_rows["artist_alias"] == artist
                ]

                festival_history = artist_history[
                    artist_history["festival"] == festival
                ]

                other_festival_history = artist_history[
                    artist_history["festival"] != festival
                ]

                if len(festival_history) > 0:
                    last_appearance_year = (
                        festival_history["year"].max()
                    )

                    years_since_appearance = (
                        year - last_appearance_year
                    )
                else:
                    years_since_appearance = -1

                # 1 if the artist appeared this year, otherwise 0
                appeared = int(artist in current_artists)

                feature_rows.append({
                    "festival": festival,
                    "year": year,
                    "artist_alias": artist,

                    "artist_total_prior_appearances": len(
                        artist_history
                    ),

                    "artist_festival_prior_appearances": len(
                        festival_history
                    ),

                    "artist_other_festival_prior_appearances": len(
                        other_festival_history
                    ),

                    "years_since_festival_appearance":
                        years_since_appearance,

                    "festival_previous_year_size": len(
                        previous_year_rows
                    ),

                    "appeared": appeared,
                })

    features = pd.DataFrame(feature_rows)

    features = features.sort_values(
        by=["festival", "year", "artist_alias"]
    )

    features.to_csv(output_path, index=False)

    return features


if __name__ == "__main__":
    create_historical_features(
        "data/processed/lineups_clean.csv",
        "data/processed/features.csv"
    )