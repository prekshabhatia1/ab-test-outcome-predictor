
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.bayesian_model import (
    calculate_posterior,
    estimate_probability_better,
    calculate_credible_interval,
    calculate_uplift,
)


app = FastAPI(
    title="Bayesian A/B Test Outcome Predictor",
    description="Bayesian A/B testing inference API deployed with FastAPI and Docker.",
    version="1.0.0",
)


class ABTestRequest(BaseModel):
    ad_users: int = Field(..., gt=0)
    ad_conversions: int = Field(..., ge=0)
    psa_users: int = Field(..., gt=0)
    psa_conversions: int = Field(..., ge=0)


def run_prediction(request: ABTestRequest):

    if request.ad_conversions > request.ad_users:
        raise HTTPException(
            status_code=400,
            detail="ad_conversions cannot be greater than ad_users.",
        )

    if request.psa_conversions > request.psa_users:
        raise HTTPException(
            status_code=400,
            detail="psa_conversions cannot be greater than psa_users.",
        )

    if request.psa_conversions == 0:
        raise HTTPException(
            status_code=400,
            detail="psa_conversions must be greater than 0 to calculate relative uplift.",
        )

    ad_posterior = calculate_posterior(
        request.ad_users,
        request.ad_conversions,
    )

    psa_posterior = calculate_posterior(
        request.psa_users,
        request.psa_conversions,
    )

    probability_ad_better = estimate_probability_better(
        ad_posterior,
        psa_posterior,
    )

    ad_lower, ad_upper = calculate_credible_interval(ad_posterior)

    psa_lower, psa_upper = calculate_credible_interval(psa_posterior)

    ad_rate = request.ad_conversions / request.ad_users
    psa_rate = request.psa_conversions / request.psa_users

    absolute_uplift, relative_uplift = calculate_uplift(
        ad_rate,
        psa_rate,
    )

    return {
        "treatment": {
            "users": request.ad_users,
            "conversions": request.ad_conversions,
            "conversion_rate": ad_rate,
            "credible_interval_95": [
                ad_lower,
                ad_upper,
            ],
        },
        "control": {
            "users": request.psa_users,
            "conversions": request.psa_conversions,
            "conversion_rate": psa_rate,
            "credible_interval_95": [
                psa_lower,
                psa_upper,
            ],
        },
        "bayesian_result": {
            "probability_treatment_better": probability_ad_better,
        },
        "treatment_effect": {
            "absolute_uplift": absolute_uplift,
            "relative_uplift": relative_uplift,
        },
    }


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/invocations")
def invocations(request: ABTestRequest):
    return run_prediction(request)


@app.post("/predict")
def predict(request: ABTestRequest):
    return run_prediction(request)
