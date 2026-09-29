import gradio as gr

from src.bayesian_model import (
    calculate_posterior,
    estimate_probability_better,
    calculate_credible_interval,
    calculate_uplift,
)


def run_ab_test(
    treatment_users,
    treatment_conversions,
    control_users,
    control_conversions,
):
    # Basic validation
    if treatment_users <= 0 or control_users <= 0:
        return "Users must be greater than 0."

    if treatment_conversions < 0 or control_conversions < 0:
        return "Conversions cannot be negative."

    if treatment_conversions > treatment_users:
        return "Treatment conversions cannot exceed treatment users."

    if control_conversions > control_users:
        return "Control conversions cannot exceed control users."

    treatment_posterior = calculate_posterior(
        int(treatment_users),
        int(treatment_conversions)
    )

    control_posterior = calculate_posterior(
        int(control_users),
        int(control_conversions)
    )

    probability_treatment_better = estimate_probability_better(
        treatment_posterior,
        control_posterior
    )

    treatment_lower, treatment_upper = calculate_credible_interval(
        treatment_posterior
    )

    control_lower, control_upper = calculate_credible_interval(
        control_posterior
    )

    treatment_rate = treatment_conversions / treatment_users
    control_rate = control_conversions / control_users

    absolute_uplift, relative_uplift = calculate_uplift(
        treatment_rate,
        control_rate
    )

    result = f"""
## Bayesian A/B Test Result

### Conversion Rates

**Treatment:** {treatment_rate:.2%}

95% Credible Interval:
`{treatment_lower:.2%} – {treatment_upper:.2%}`

**Control:** {control_rate:.2%}

95% Credible Interval:
`{control_lower:.2%} – {control_upper:.2%}`

---

### Bayesian Comparison

**Probability that Treatment is better:**  
### {probability_treatment_better:.2%}

---

### Treatment Effect

**Absolute Uplift:** {absolute_uplift:.2%}

**Relative Uplift:** {relative_uplift:.2%}
"""

    return result


demo = gr.Interface(
    fn=run_ab_test,
    inputs=[
        gr.Number(
            label="Treatment Users",
            value=10000,
            precision=0
        ),
        gr.Number(
            label="Treatment Conversions",
            value=1200,
            precision=0
        ),
        gr.Number(
            label="Control Users",
            value=10000,
            precision=0
        ),
        gr.Number(
            label="Control Conversions",
            value=1000,
            precision=0
        ),
    ],
    outputs=gr.Markdown(),
    title="Bayesian A/B Test Outcome Predictor",
    description=(
        "Estimate treatment effectiveness using Bayesian "
        "Beta-Binomial inference."
    ),
)

if __name__ == "__main__":
    demo.launch()