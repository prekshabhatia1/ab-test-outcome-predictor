import matplotlib.pyplot as plt

from src.early_prediction import load_data
from src.evaluate_stopping import run_multiple_simulations


def plot_stopping_distribution(results):
    """
    Plot the distribution of stopping points
    across repeated simulated experiments.
    """

    stopping_points = results[
        results["stopped"] == True
    ]["stopping_fraction"] * 100

    plt.figure(figsize=(10, 6))

    plt.hist(
        stopping_points,
        bins=range(8, 26),
        edgecolor="black"
    )

    plt.xlabel("Stopping Point (% of Experiment Data)")
    plt.ylabel("Number of Simulations")

    plt.title(
        "Distribution of Bayesian Early-Stopping Points"
    )

    plt.xticks(range(9, 25))

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    df = load_data()

    results, full_reference = run_multiple_simulations(
        df,
        number_of_simulations=100
    )

    plot_stopping_distribution(results)