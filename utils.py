UNKNOWN_GENRE = "(no genres listed)"

def preprocess_genres(df, genre_col="genres"):
    new_df = df.copy()
    new_df[genre_col] = new_df[genre_col].map(lambda genre: None if genre == UNKNOWN_GENRE else genre.split("|"))
    return new_df

