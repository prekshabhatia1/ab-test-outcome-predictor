import pandas as pd
import numpy as np

from src.early_prediction import (
    load_data,
    calculate_posterior,
    probability_ad_better,
    make_stopping_decision
)


def calculate_full_data_reference(df):
    """
    Calculate the Bayesian result using the complete dataset.
    This is used as a reference for evaluating early decisions.
    """

    ad_data = df[df["test_group"] == "ad"]
    psa_data = df[df["test_group"] == "psa"]

    ad_users = len(ad_data)
    ad_conversions = ad_data["converted"].sum()

    psa_users = len(psa_data)
    psa_conversions = psa_data["converted"].sum()

    ad_posterior = calculate_posterior(
        ad_users,
        ad_conversions
    )

    psa_posterior = calculate_posterior(
        psa_users,
        psa_conversions
    )

    probability = probability_ad_better(
        ad_posterior,
        psa_posterior
    )

    ad_rate = ad_conversions / ad_users
    psa_rate = psa_conversions / psa_users

    relative_uplift = (
        (ad_rate - psa_rate) / psa_rate
    )

    return {
        "ad_rate": ad_rate,
        "psa_rate": psa_rate,
        "probability_ad_better": probability,
        "relative_uplift": relative_uplift
    }


def find_first_stopping_point(
    df,
    seed,
    checkpoint_step=0.01
):
    """
    Simulate an experiment and stop as soon as
    the stopping criteria are satisfied.

    This is faster than calculating all 100 checkpoints
    after the experiment has already stopped.
    """

    shuffled_df = df.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    total_users = len(shuffled_df)

    fractions = np.arange(
        checkpoint_step,
        1.01,
        checkpoint_step
    )

    for fraction in fractions:

        sample_size = int(
            total_users * fraction
        )

        sample = shuffled_df.iloc[:sample_size]

        ad_data = sample[
            sample["test_group"] == "ad"
        ]

        psa_data = sample[
            sample["test_group"] == "psa"
        ]

        ad_users = len(ad_data)
        ad_conversions = ad_data["converted"].sum()

        psa_users = len(psa_data)
        psa_conversions = psa_data["converted"].sum()

        # We cannot calculate a meaningful comparison
        # until both groups contain users.
        if ad_users == 0 or psa_users == 0:
            continue

        ad_posterior = calculate_posterior(
            ad_users,
            ad_conversions
        )

        psa_posterior = calculate_posterior(
            psa_users,
            psa_conversions
        )

        probability = probability_ad_better(
            ad_posterior,
            psa_posterior,
            samples=10000,
            seed=seed
        )

        ad_rate = ad_conversions / ad_users
        psa_rate = psa_conversions / psa_users

        decision = make_stopping_decision(
            probability_ad_better=probability,
            ad_rate=ad_rate,
            psa_rate=psa_rate,
            users=sample_size
        )

        if decision["should_stop"]:

            return {
                "stopped": True,
                "stopping_fraction": fraction,
                "users_at_stop": sample_size,
                "probability_ad_better": probability,
                "relative_uplift": decision["relative_uplift"]
            }

    return {
        "stopped": False,
        "stopping_fraction": np.nan,
        "users_at_stop": np.nan,
        "probability_ad_better": np.nan,
        "relative_uplift": np.nan
    }


def run_multiple_simulations(
    df,
    number_of_simulations=100
):
    """
    Run the early-stopping simulation multiple times
    using different random seeds.
    """

    full_reference = calculate_full_data_reference(df)

    simulation_results = []

    for simulation_number in range(number_of_simulations):

        seed = 42 + simulation_number

        print(
            f"Running simulation "
            f"{simulation_number + 1}/{number_of_simulations}..."
        )

        result = find_first_stopping_point(
            df,
            seed=seed
        )

        if result["stopped"]:

            early_direction = (
                result["probability_ad_better"] >= 0.50
            )

            full_direction = (
                full_reference["probability_ad_better"] >= 0.50
            )

            decision_agreement = (
                early_direction == full_direction
            )

            uplift_error = abs(
                result["relative_uplift"]
                - full_reference["relative_uplift"]
            )

            simulation_results.append({
                "simulation": simulation_number + 1,
                "seed": seed,
                "stopped": True,
                "stopping_fraction": result["stopping_fraction"],
                "users_at_stop": result["users_at_stop"],
                "probability_ad_better": result["probability_ad_better"],
                "relative_uplift": result["relative_uplift"],
                "decision_agreement": decision_agreement,
                "uplift_error": uplift_error
            })

        else:

            simulation_results.append({
                "simulation": simulation_number + 1,
                "seed": seed,
                "stopped": False,
                "stopping_fraction": np.nan,
                "users_at_stop": np.nan,
                "probability_ad_better": np.nan,
                "relative_uplift": np.nan,
                "decision_agreement": False,
                "uplift_error": np.nan
            })

    return (
        pd.DataFrame(simulation_results),
        full_reference
    )


if __name__ == "__main__":

    df = load_data()

    results, full_reference = run_multiple_simulations(
        df,
        number_of_simulations=100
    )

    print("\nFull-Data Reference")
    print("=" * 60)

    print(
        f"Ad conversion rate: "
        f"{full_reference['ad_rate']:.4%}"
    )

    print(
        f"PSA conversion rate: "
        f"{full_reference['psa_rate']:.4%}"
    )

    print(
        f"P(Ad > PSA): "
        f"{full_reference['probability_ad_better']:.2%}"
    )

    print(
        f"Relative uplift: "
        f"{full_reference['relative_uplift']:.2%}"
    )

    print("\nRepeated Early-Stopping Evaluation")
    print("=" * 80)

    print(
        results[
            [
                "simulation",
                "stopping_fraction",
                "users_at_stop",
                "probability_ad_better",
                "relative_uplift",
                "decision_agreement",
                "uplift_error"
            ]
        ].to_string(index=False)
    )

    stopped_results = results[
        results["stopped"] == True
    ]

    print("\nEvaluation Summary")
    print("=" * 60)

    print(
        f"Simulations: "
        f"{len(results)}"
    )

    print(
        f"Experiments that stopped: "
        f"{len(stopped_results)}"
    )

    print(
        f"Stopping rate: "
        f"{len(stopped_results) / len(results):.2%}"
    )

    if len(stopped_results) > 0:

        print(
            f"Average stopping point: "
            f"{stopped_results['stopping_fraction'].mean():.2%}"
        )

        print(
            f"Median stopping point: "
            f"{stopped_results['stopping_fraction'].median():.2%}"
        )

        print(
            f"Minimum stopping point: "
            f"{stopped_results['stopping_fraction'].min():.2%}"
        )

        print(
            f"Maximum stopping point: "
            f"{stopped_results['stopping_fraction'].max():.2%}"
        )

        print(
            f"Average users at stop: "
            f"{stopped_results['users_at_stop'].mean():,.0f}"
        )

        print(
            f"Decision agreement with full data: "
            f"{stopped_results['decision_agreement'].mean():.2%}"
        )

        print(
            f"Mean uplift error: "
            f"{stopped_results['uplift_error'].mean():.2%}"
        )

        print(
            f"Median uplift error: "
            f"{stopped_results['uplift_error'].median():.2%}"
        )