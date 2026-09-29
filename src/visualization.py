import matplotlib.pyplot as plt

from src.early_prediction import (
    load_data,
    simulate_partial_experiment
)


def plot_conversion_rates(results):
    """
    Plot Ad and PSA conversion rates
    with their 95% Bayesian credible intervals.
    """

    # Convert fractions to percentages
    x = results["fraction"] * 100

    # Convert conversion rates to percentages
    ad_rate = results["ad_rate"] * 100
    psa_rate = results["psa_rate"] * 100

    # Convert credible interval bounds to percentages
    ad_lower = results["ad_lower"] * 100
    ad_upper = results["ad_upper"] * 100

    psa_lower = results["psa_lower"] * 100
    psa_upper = results["psa_upper"] * 100

    # Calculate error bar sizes
    ad_error_lower = ad_rate - ad_lower
    ad_error_upper = ad_upper - ad_rate

    psa_error_lower = psa_rate - psa_lower
    psa_error_upper = psa_upper - psa_rate

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot Ad / Treatment
    plt.errorbar(
        x,
        ad_rate,
        yerr=[ad_error_lower, ad_error_upper],
        marker="o",
        capsize=5,
        label="Ad (Treatment)"
    )

    # Plot PSA / Control
    plt.errorbar(
        x,
        psa_rate,
        yerr=[psa_error_lower, psa_error_upper],
        marker="o",
        capsize=5,
        label="PSA (Control)"
    )

    # Labels
    plt.xlabel("Percentage of Experiment Data Available")
    plt.ylabel("Conversion Rate (%)")

    # Title
    plt.title(
        "Bayesian A/B Test: Conversion Rate and 95% Credible Intervals"
    )

    # Show all fraction points
    plt.xticks(x)

    # Grid
    plt.grid(True, alpha=0.3)

    # Legend
    plt.legend()

    # Improve spacing
    plt.tight_layout()

    # Display plot
    plt.show()


def plot_probability_ad_better(results):
    """
    Plot the posterior probability that
    Ad performs better than PSA.
    """

    # Convert fraction to percentage
    x = results["fraction"] * 100

    # Convert probability to percentage
    probability = results["probability_ad_better"] * 100

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot probability
    plt.plot(
        x,
        probability,
        marker="o",
        label="P(Ad > PSA)"
    )

    # Reference line at 95%
    plt.axhline(
        y=95,
        linestyle="--",
        label="95% Threshold"
    )

    # Labels
    plt.xlabel("Percentage of Experiment Data Available")
    plt.ylabel("Probability Ad > PSA (%)")

    # Title
    plt.title(
        "Bayesian Evidence as More Experiment Data Arrives"
    )

    # Show all fraction points
    plt.xticks(x)

    # Probability ranges from 0 to 100
    plt.ylim(0, 101)

    # Grid
    plt.grid(True, alpha=0.3)

    # Legend
    plt.legend()

    # Improve spacing
    plt.tight_layout()

    # Display plot
    plt.show()


if __name__ == "__main__":

    # Load dataset
    df = load_data()

    # Run early experiment simulation
    results = simulate_partial_experiment(df)

    # Generate conversion rate plot
    plot_conversion_rates(results)

    # Generate probability plot
    plot_probability_ad_better(results)