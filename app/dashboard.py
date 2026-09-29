import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.bayesian_model import (
    load_data,
    calculate_group_statistics,
    calculate_posterior,
    estimate_probability_better,
    calculate_credible_interval,
    calculate_uplift
)

from src.early_prediction import (
    simulate_partial_experiment
)


st.set_page_config(
    page_title="Bayesian A/B Test Predictor",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Bayesian A/B Test Outcome Predictor")

st.write(
    """
    Analyze an A/B experiment using Bayesian inference
    and estimate whether the treatment is performing
    better than the control group.
    """
)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = load_data()

statistics = calculate_group_statistics(df)


# --------------------------------------------------
# Group Statistics
# --------------------------------------------------

ad_users = statistics["ad"]["users"]
ad_conversions = statistics["ad"]["conversions"]
ad_rate = statistics["ad"]["conversion_rate"]

psa_users = statistics["psa"]["users"]
psa_conversions = statistics["psa"]["conversions"]
psa_rate = statistics["psa"]["conversion_rate"]


# --------------------------------------------------
# Bayesian Model
# --------------------------------------------------

ad_posterior = calculate_posterior(
    ad_users,
    ad_conversions
)

psa_posterior = calculate_posterior(
    psa_users,
    psa_conversions
)


probability_ad_better = estimate_probability_better(
    ad_posterior,
    psa_posterior
)


ad_lower, ad_upper = calculate_credible_interval(
    ad_posterior
)

psa_lower, psa_upper = calculate_credible_interval(
    psa_posterior
)


absolute_uplift, relative_uplift = calculate_uplift(
    ad_rate,
    psa_rate
)


# --------------------------------------------------
# Experiment Overview
# --------------------------------------------------

st.header("Experiment Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Total Users",
        f"{len(df):,}"
    )


with col2:
    st.metric(
        "Ad Users",
        f"{ad_users:,}"
    )


with col3:
    st.metric(
        "PSA Users",
        f"{psa_users:,}"
    )


with col4:
    st.metric(
        "Ad Conversion Rate",
        f"{ad_rate:.2%}"
    )


# --------------------------------------------------
# Conversion Rates
# --------------------------------------------------

st.header("Conversion Rates")


col1, col2 = st.columns(2)


with col1:

    st.subheader("Ad — Treatment")

    st.metric(
        "Conversion Rate",
        f"{ad_rate:.2%}"
    )

    st.write(
        f"Conversions: {ad_conversions:,}"
    )

    st.write(
        f"95% Credible Interval: "
        f"{ad_lower:.2%} – {ad_upper:.2%}"
    )


with col2:

    st.subheader("PSA — Control")

    st.metric(
        "Conversion Rate",
        f"{psa_rate:.2%}"
    )

    st.write(
        f"Conversions: {psa_conversions:,}"
    )

    st.write(
        f"95% Credible Interval: "
        f"{psa_lower:.2%} – {psa_upper:.2%}"
    )


# --------------------------------------------------
# Bayesian Result
# --------------------------------------------------

st.header("Bayesian Result")


st.metric(
    "Probability that Ad > PSA",
    f"{probability_ad_better:.2%}"
)


st.write(
    """
    This represents the posterior probability that the
    underlying conversion rate of the Ad treatment is
    higher than the PSA control under the specified
    Bayesian model and prior.
    """
)


# --------------------------------------------------
# Treatment Effect
# --------------------------------------------------

st.header("Treatment Effect")


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Absolute Uplift",
        f"{absolute_uplift:.2%}"
    )


with col2:

    st.metric(
        "Relative Uplift",
        f"{relative_uplift:.2%}"
    )
# --------------------------------------------------
# Early Stopping Analysis
# --------------------------------------------------

st.header("Early Stopping Analysis")

st.write(
    """
    This simulation evaluates how early the experiment
    could satisfy the predefined stopping criteria.
    """
)

with st.spinner("Running early-stopping simulation..."):

    early_results = simulate_partial_experiment(df)


# Find first stopping point

stopping_points = early_results[
    early_results["should_stop"] == True
]


if len(stopping_points) > 0:

    first_stop = stopping_points.iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "First Stopping Point",
            f"{first_stop['fraction']:.0%}"
        )

    with col2:
        st.metric(
            "Users at Stop",
            f"{first_stop['users']:,}"
        )

    with col3:
        st.metric(
            "P(Ad > PSA)",
            f"{first_stop['probability_ad_better']:.2%}"
        )

    st.success(
        "The predefined stopping criteria were satisfied."
    )

else:

    st.warning(
        "The predefined stopping criteria were not satisfied."
    )


# --------------------------------------------------
# Probability Chart
# --------------------------------------------------

st.subheader("Probability That Ad Outperforms PSA")

chart_data = early_results[
    [
        "fraction",
        "probability_ad_better"
    ]
].copy()

chart_data["fraction"] = (
    chart_data["fraction"] * 100
)

chart_data = chart_data.set_index("fraction")

st.line_chart(
    chart_data
)


# --------------------------------------------------
# Early Stopping Analysis
# --------------------------------------------------

st.header("Early Stopping Analysis")

st.write(
    """
    This simulation evaluates how early the experiment
    could satisfy the predefined stopping criteria.
    """
)

with st.spinner("Running early-stopping simulation..."):

    early_results = simulate_partial_experiment(df)


# --------------------------------------------------
# First Stopping Point
# --------------------------------------------------

stopping_points = early_results[
    early_results["should_stop"] == True
]


if len(stopping_points) > 0:

    first_stop = stopping_points.iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "First Stopping Point",
            f"{first_stop['fraction']:.0%}"
        )

    with col2:
        st.metric(
            "Users at Stop",
            f"{first_stop['users']:,}"
        )

    with col3:
        st.metric(
            "P(Ad > PSA)",
            f"{first_stop['probability_ad_better']:.2%}"
        )

    st.success(
        "The predefined stopping criteria were satisfied."
    )

else:

    st.warning(
        "The predefined stopping criteria were not satisfied."
    )


# --------------------------------------------------
# Probability Chart
# --------------------------------------------------

st.subheader(
    "Probability That Ad Outperforms PSA"
)

chart_data = early_results[
    [
        "fraction",
        "probability_ad_better"
    ]
].copy()

chart_data["fraction"] = (
    chart_data["fraction"] * 100
)

chart_data = chart_data.set_index("fraction")

st.line_chart(
    chart_data
)


# --------------------------------------------------
# Stopping Criteria
# --------------------------------------------------

st.subheader("Stopping Criteria")

st.write(
    """
    The experiment is considered ready to stop when all
    three conditions are satisfied:
    """
)

st.markdown(
    """
    - **Posterior probability ≥ 95%**
    - **At least 50,000 users observed**
    - **Relative uplift ≥ 10%**
    """
)


# --------------------------------------------------
# Repeated Simulation Distribution
# --------------------------------------------------

st.subheader(
    "Distribution of Early-Stopping Points"
)

st.write(
    """
    The chart below shows where the experiment stopped
    across 100 different simulated experiments.
    """
)

from src.evaluate_stopping import (
    run_multiple_simulations
)


@st.cache_data
def get_simulation_results(df):

    return run_multiple_simulations(
        df,
        number_of_simulations=100
    )


with st.spinner(
    "Running repeated stopping simulations..."
):

    simulation_results, full_reference = (
        get_simulation_results(df)
    )


stopping_distribution = (
    simulation_results[
        simulation_results["stopped"] == True
    ]["stopping_fraction"] * 100
).value_counts().sort_index()


st.bar_chart(
    stopping_distribution
)


# --------------------------------------------------
# Evaluation Summary
# --------------------------------------------------

st.subheader("Repeated Simulation Summary")

stopping_results = simulation_results[
    simulation_results["stopped"] == True
]


if len(stopping_results) > 0:

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Experiments Stopped",
            f"{len(stopping_results)}/100"
        )

    with col2:
        st.metric(
            "Median Stopping Point",
            f"{stopping_results['stopping_fraction'].median():.0%}"
        )

    with col3:
        st.metric(
            "Average Stopping Point",
            f"{stopping_results['stopping_fraction'].mean():.2%}"
        )

    with col4:
        st.metric(
            "Decision Agreement",
            f"{stopping_results['decision_agreement'].mean():.2%}"
        )