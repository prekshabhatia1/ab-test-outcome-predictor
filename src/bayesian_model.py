
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "ab_project_marketing_events_us.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.rename(
        columns={
            "user id": "user_id",
            "test group": "test_group",
            "total ads": "total_ads",
            "most ads day": "most_ads_day",
            "most ads hour": "most_ads_hour",
        }
    )

    return df


def calculate_group_statistics(df):

    statistics = {}

    for group in ["ad", "psa"]:

        group_data = df[df["test_group"] == group]

        users = len(group_data)
        conversions = group_data["converted"].sum()
        conversion_rate = conversions / users

        statistics[group] = {
            "users": users,
            "conversions": conversions,
            "conversion_rate": conversion_rate,
        }

    return statistics


def calculate_posterior(
    users,
    conversions,
    alpha_prior=1,
    beta_prior=1,
):

    failures = users - conversions

    posterior_alpha = alpha_prior + conversions
    posterior_beta = beta_prior + failures

    return beta(
        posterior_alpha,
        posterior_beta,
    )


def calculate_credible_interval(
    posterior,
    confidence=0.95,
):

    lower_probability = (1 - confidence) / 2
    upper_probability = 1 - lower_probability

    lower = posterior.ppf(lower_probability)
    upper = posterior.ppf(upper_probability)

    return lower, upper


def calculate_uplift(ad_rate, psa_rate):

    absolute_uplift = ad_rate - psa_rate

    relative_uplift = (
        (ad_rate - psa_rate) / psa_rate
    )

    return absolute_uplift, relative_uplift


def estimate_probability_better(
    posterior_ad,
    posterior_psa,
    samples=100_000,
    seed=42,
):

    rng = np.random.default_rng(seed)

    ad_samples = posterior_ad.rvs(
        size=samples,
        random_state=rng,
    )

    psa_samples = posterior_psa.rvs(
        size=samples,
        random_state=rng,
    )

    probability = np.mean(
        ad_samples > psa_samples
    )

    return float(probability)


if __name__ == "__main__":

    df = load_data()

    statistics = calculate_group_statistics(df)

    ad_posterior = calculate_posterior(
        statistics["ad"]["users"],
        statistics["ad"]["conversions"],
    )

    psa_posterior = calculate_posterior(
        statistics["psa"]["users"],
        statistics["psa"]["conversions"],
    )

    probability_ad_better = estimate_probability_better(
        ad_posterior,
        psa_posterior,
    )

    ad_lower, ad_upper = calculate_credible_interval(
        ad_posterior,
    )

    psa_lower, psa_upper = calculate_credible_interval(
        psa_posterior,
    )

    ad_rate = statistics["ad"]["conversion_rate"]
    psa_rate = statistics["psa"]["conversion_rate"]

    absolute_uplift, relative_uplift = calculate_uplift(
        ad_rate,
        psa_rate,
    )

    print("\nBayesian A/B Test")
    print("=" * 40)

    print("\nAd / Treatment")
    print(f"Observed conversion rate: {ad_rate:.4%}")
    print(
        f"95% credible interval: "
        f"[{ad_lower:.4%}, {ad_upper:.4%}]"
    )

    print("\nPSA / Control")
    print(f"Observed conversion rate: {psa_rate:.4%}")
    print(
        f"95% credible interval: "
        f"[{psa_lower:.4%}, {psa_upper:.4%}]"
    )

    print("\nTreatment Effect")
    print(f"Absolute uplift: {absolute_uplift:.4%}")
    print(f"Relative uplift: {relative_uplift:.2%}")

    print(
        "\nProbability that Ad > PSA: "
        f"{probability_ad_better:.4%}"
    )

