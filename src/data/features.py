"""Feature Engineering

Create a new csv with metrics on past appearances and negative examples for model training"""

import pandas as pd

def create_historical_features(input_path, output_path):
    df = pd.read_csv(input_path)
    feature_rows = []
    #Pick out festivals available (will be more useful with more festivals in the database)
    festivals = df["festival"].unique()
    #Mark out the available years in the data
    years = sorted(df["year"].unique())

    for festival in festivals:
        for year in years:
            #Do not include previous years for negative examples
            previous_rows = df[df["year"] < year]

            #Set of artists who have already performed in prior years
            previous_artists = set(
                previous_rows["artist_alias"]
            )

            #Show artists who appeared in the current year
            current_artists = set(
                df[
                    (df["festival"] == festival) &
                    (df["year"] == year)
                ]["artist_alias"]
            )
            #List of possible artists available for negative examples
            candidate_artists = previous_artists.union(current_artists)

            previous_year_rows = df[
                (df["festival"] == festival) &
                (df["year"] == year - 1)
            ]
            #Mark artists in complete data frame
            #1 is appeared and 0 is did not appear
            #Tally up totals for each artists' previous appearances in each category
            for artist in candidate_artists:
                #How many times have they appeared
                artist_history = previous_rows[
                    previous_rows["artist_alias"] == artist
                ]
                #How many times have they appeared at that specific festival
                festival_history = artist_history[
                    artist_history["festival"] == festival
                ]
                #How many times they appeared at other festivals
                other_festival_history = artist_history[
                    artist_history["festival"] != festival
                ]
                #Artist has never appeared
                if len(festival_history) > 0:
                    last_appearance_year = (
                        festival_history["year"].max()
                    )

                    years_since_appearance = (
                        year - last_appearance_year
                    )
                else:
                    years_since_appearance = -1
        
                appeared = int(artist in current_artists)
                #Add each row into the feature rows data frame
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
    #Save final to the processed data folder
    features.to_csv(output_path, index=False)
    return features


if __name__ == "__main__":
    create_historical_features(
        "data/processed/lineups_clean.csv",
        "data/processed/features.csv"
    )