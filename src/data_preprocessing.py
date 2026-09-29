import pandas as pd


DATA_PATH = "data/ab_project_marketing_events_us.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    return df


def clean_data(df):
    # Remove the unnecessary CSV index column
    df = df.drop(columns=["Unnamed: 0"])

    # Remove duplicate users if any
    df = df.drop_duplicates(subset=["user id"])

    # Make column names easier to work with
    df = df.rename(columns={
        "user id": "user_id",
        "test group": "test_group",
        "total ads": "total_ads",
        "most ads day": "most_ads_day",
        "most ads hour": "most_ads_hour"
    })

    return df


if __name__ == "__main__":

    df = load_data()

    print("Original shape:", df.shape)

    df = clean_data(df)

    print("Cleaned shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDuplicate users:")
    print(df["user_id"].duplicated().sum())

    print("\nTest groups:")
    print(df["test_group"].value_counts())

    print("\nConversion rate by group:")
    print(
        df.groupby("test_group")["converted"].mean()
    )