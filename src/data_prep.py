import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def load_data(data_path):
    """
    Load the Walmart dataset.
    """
    df = pd.read_csv(data_path)
    print("Dataset shape:", df.shape)
    print(df.head())
    return df


def inspect_data(df):
    """
    Print a compact overview of the dataset.
    """
    print("\nMissing values by column:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.info())

    print("\nCategorical summary:")
    print(df.describe(include="object").T)

    print("\nPurchase summary:")
    print(df["Purchase"].describe())


def cast_columns(df):
    """
    Convert columns that represent categories into object dtype
    for cleaner exploratory analysis.
    """
    df = df.copy()

    df["User_ID"] = df["User_ID"].astype("object")
    df["Occupation"] = df["Occupation"].astype("object")
    df["Product_Category"] = df["Product_Category"].astype("object")
    df["Marital_Status"] = df["Marital_Status"].astype("object")

    return df


def print_unique_values(df):
    """
    Print unique values for selected categorical features.
    """
    selected_columns = ["Age", "City_Category", "Occupation", "Stay_In_Current_City_Years"]

    for col in selected_columns:
        print(f"\nUnique values of '{col}':")
        print(df[col].unique())


def plot_categorical_distributions(df, save_path=None):
    """
    Plot the distribution of key categorical features.
    """
    categorical_vars = ["Age", "Marital_Status", "City_Category", "Gender"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, var in zip(axes.flatten(), categorical_vars):
        sns.countplot(x=df[var], order=df[var].value_counts().index, ax=ax)
        ax.set_title(f"Distribution of {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Count")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def prepare_features(df):
    """
    Transform the dataset into a feature set suitable for clustering.

    Steps:
    - Encode Gender as binary
    - Map Age ranges to ordinal values and scale them
    - One-hot encode City_Category
    """
    df = df.copy()

    # Binary encoding
    df["Gender"] = df["Gender"].map({"M": 1, "F": 0})

    # Ordinal mapping for age groups
    age_mapping = {
        "0-17": 0,
        "18-25": 1,
        "26-35": 2,
        "36-45": 3,
        "46-50": 4,
        "51-55": 5,
        "55+": 6
    }
    df["Age"] = df["Age"].map(age_mapping)

    # Scale age to [0, 1]
    scaler = MinMaxScaler()
    df["Age"] = scaler.fit_transform(df[["Age"]])

    # One-hot encode city category
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    city_encoded = encoder.fit_transform(df[["City_Category"]])

    city_df = pd.DataFrame(
        city_encoded,
        columns=encoder.get_feature_names_out(["City_Category"]),
        index=df.index
    )

    df = pd.concat([df, city_df], axis=1)
    df.drop(columns=["City_Category"], inplace=True)

    # Make sure Marital_Status is numeric for clustering
    df["Marital_Status"] = df["Marital_Status"].astype(int)

    return df


def get_clustering_data(df):
    """
    Return the feature matrix and a DataFrame version of the same
    features used for clustering.
    """
    feature_columns = [
        "Age",
        "Marital_Status",
        "Gender",
        "City_Category_A",
        "City_Category_B",
        "City_Category_C",
    ]

    data_df = df[feature_columns].copy()
    data = data_df.values

    return data, data_df

