import pandas as pd
import numpy as np
from scipy.stats import beta


DATA_PATH = "data/ab_project_marketing_events_us.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df.drop(columns=["Unnamed: 0"])

    df = df.rename(columns={
        "user id": "user_id",
        "test group": "test_group",
        "total ads": "total_ads",
        "most ads day": "most_ads_day",
        "most ads hour": "most_ads_hour"
    })

    return df


def calculate_posterior(
    users,
    conversions,
    alpha_prior=1,
    beta_prior=1
):
    """
    Calculate the Beta posterior distribution
    for a group's conversion rate.
    """

    failures = users - conversions

    alpha = alpha_prior + conversions
    beta_parameter = beta_prior + failures

    return beta(alpha, beta_parameter)


def probability_ad_better(
    ad_posterior,
    psa_posterior,
    samples=50000,
    seed=42
):
    """
    Estimate the posterior probability that
    the Ad group's conversion rate is higher
    than the PSA group's conversion rate.
    """

    rng = np.random.default_rng(seed)

    ad_samples = ad_posterior.rvs(
        size=samples,
        random_state=rng
    )

    psa_samples = psa_posterior.rvs(
        size=samples,
        random_state=rng
    )

    return np.mean(ad_samples > psa_samples)


def calculate_credible_interval(
    posterior,
    confidence=0.95
):
    """
    Calculate a Bayesian credible interval.
    """

    lower_probability = (1 - confidence) / 2
    upper_probability = 1 - lower_probability

    lower = posterior.ppf(lower_probability)
    upper = posterior.ppf(upper_probability)

    return lower, upper


def make_stopping_decision(
   
    probability_ad_better,
    ad_rate,
    psa_rate,
    users,
    probability_threshold=0.95,
    minimum_users=50000,
    minimum_relative_uplift=0.10
):
    """
    Decide whether the simulated experiment
    satisfies all early-stopping criteria.

    Conditions:

    1. Posterior probability >= 95%
    2. At least 50,000 users observed
    3. Relative uplift >= 10%
    """

    # Relative uplift cannot be calculated
    # when the control conversion rate is zero.
    if psa_rate == 0:
        relative_uplift = np.inf
    else:
        relative_uplift = (
            (ad_rate - psa_rate) / psa_rate
        )

    probability_condition = (
        probability_ad_better >= probability_threshold
    )

    sample_size_condition = (
        users >= minimum_users
    )

    uplift_condition = (
        relative_uplift >= minimum_relative_uplift
    )

    should_stop = (
        probability_condition
        and sample_size_condition
        and uplift_condition
    )

    return {
        "should_stop": should_stop,
        "relative_uplift": relative_uplift,
        "probability_condition": probability_condition,
        "sample_size_condition": sample_size_condition,
        "uplift_condition": uplift_condition
    }


def simulate_partial_experiment(
    df,
    fractions=np.arange(0.01, 1.01, 0.01),
    seed=42
):
    """
    Simulate an experiment where increasing
    fractions of the dataset become available.

    Because the dataset does not contain experiment
    timestamps, rows are randomly shuffled first.
    """

    # Randomize row order
    shuffled_df = df.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    results = []

    for fraction in fractions:

        # Number of users available at this stage
        sample_size = int(
            len(shuffled_df) * fraction
        )

        sample = shuffled_df.iloc[:sample_size]

        # Treatment group
        ad_data = sample[
            sample["test_group"] == "ad"
        ]

        # Control group
        psa_data = sample[
            sample["test_group"] == "psa"
        ]

        # Treatment statistics
        ad_users = len(ad_data)
        ad_conversions = ad_data["converted"].sum()

        # Control statistics
        psa_users = len(psa_data)
        psa_conversions = psa_data["converted"].sum()

        # Bayesian posterior for Ad
        ad_posterior = calculate_posterior(
            ad_users,
            ad_conversions
        )

        # Bayesian posterior for PSA
        psa_posterior = calculate_posterior(
            psa_users,
            psa_conversions
        )

        # Probability that Ad is better
        probability = probability_ad_better(
            ad_posterior,
            psa_posterior,
            seed=seed
        )

        # Observed conversion rates
        ad_rate = ad_conversions / ad_users
        psa_rate = psa_conversions / psa_users

        # 95% credible intervals
        ad_lower, ad_upper = calculate_credible_interval(
            ad_posterior
        )

        psa_lower, psa_upper = calculate_credible_interval(
            psa_posterior
        )

        # Early stopping decision
        decision = make_stopping_decision(
            probability_ad_better=probability,
            ad_rate=ad_rate,
            psa_rate=psa_rate,
            users=sample_size
        )

        # Store all results
        results.append({
            "fraction": fraction,
            "users": sample_size,
            "ad_users": ad_users,
            "psa_users": psa_users,
            "ad_rate": ad_rate,
            "psa_rate": psa_rate,
            "ad_lower": ad_lower,
            "ad_upper": ad_upper,
            "psa_lower": psa_lower,
            "psa_upper": psa_upper,
            "relative_uplift": decision["relative_uplift"],
            "probability_ad_better": probability,
            "should_stop": decision["should_stop"]
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    # Load dataset
    df = load_data()

    # Run early experiment simulation
    results = simulate_partial_experiment(df)

    # Display results
    print("\nEarly Experiment Simulation")
    print("=" * 80)

    print(
        results[
            [
                "fraction",
                "users",
                "ad_rate",
                "psa_rate",
                "relative_uplift",
                "probability_ad_better",
                "should_stop"
            ]
        ].to_string(index=False)
    )

    # Find the first point where all
    # stopping criteria are satisfied
    stopping_points = results[
        results["should_stop"] == True
    ]

    if len(stopping_points) > 0:

        first_stop = stopping_points.iloc[0]

        print("\nFirst Stopping Point")
        print("=" * 50)

        print(
            f"Data available: "
            f"{first_stop['fraction']:.0%}"
        )

        print(
            f"Users observed: "
            f"{first_stop['users']:,}"
        )

        print(
            f"P(Ad > PSA): "
            f"{first_stop['probability_ad_better']:.2%}"
        )

        print(
            f"Relative uplift: "
            f"{first_stop['relative_uplift']:.2%}"
        )

    else:

        print("\nNo stopping point reached.")