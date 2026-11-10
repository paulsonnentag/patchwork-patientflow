"""Emergency bed-demand analysis on the patientflow synthetic data.

The question: at a given time of day, how many beds will emergency admissions
need over the next few hours? Demand comes from patients already in the ED
(per-patient classifier, aggregated) plus patients yet to arrive (arrival
rates, assuming the ED meets its admission target).
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from patientflow.aggregate import get_prob_dist
from patientflow.load import get_model_key, load_config_file, load_data
from patientflow.predictors.incoming_admission_predictors import (
    ParametricIncomingAdmissionPredictor,
)
from patientflow.prepare import (
    create_temporal_splits,
    prepare_group_snapshot_dict,
    prepare_patient_snapshots,
)
from patientflow.train.classifiers import TrainedClassifier, train_classifier
from patientflow.viz.epudd import plot_epudd
from patientflow.viz.madcap import plot_madcap
from patientflow.viz.probability_distribution import plot_prob_dist

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

MODEL_NAME = "admissions"
EXAMPLE_DAY = date(2031, 10, 11)

EXCLUDE_FROM_TRAINING = [
    "snapshot_date",
    "prediction_time",
    "visit_number",
    "consultation_sequence",
    "specialty",
    "final_sequence",
]
ORDINAL_MAPPINGS = {
    "age_group": ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-115"],
    "latest_obs_manchester_triage_acuity": ["Blue", "Green", "Yellow", "Orange", "Red"],
    "latest_obs_objective_pain_score": ["Nil", "Mild", "Moderate", "Severe_Very Severe"],
    "latest_obs_level_of_consciousness": ["A", "C", "V", "P", "U"],
}

PredictionTime = tuple[int, int]
ProbDists = dict[str, dict[date, dict[str, Any]]]


@dataclass
class Inputs:
    ed_visits: pd.DataFrame
    inpatient_arrivals: pd.DataFrame
    params: dict[str, Any]


@dataclass
class Splits:
    train: pd.DataFrame
    calibration: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass
class AnalysisResults:
    inputs: Inputs
    splits: Splits
    prediction_times: list[PredictionTime]
    models: dict[str, TrainedClassifier]
    prob_dists: ProbDists
    example_day: date
    prediction_time: PredictionTime
    prediction_window: timedelta
    yet_to_arrive: pd.DataFrame
    total_demand: pd.DataFrame


def run_analysis(
    example_day: date = EXAMPLE_DAY,
    prediction_time: PredictionTime = (12, 0),
    prediction_window: timedelta = timedelta(hours=8),
) -> AnalysisResults:
    inputs = load_inputs()
    splits = split_visits(inputs)
    prediction_times = sorted(inputs.ed_visits.prediction_time.unique())
    models = train_models(splits, prediction_times)
    prob_dists = predict_bed_counts(models, splits.test, prediction_times)

    in_ed = prob_dists[get_model_key(MODEL_NAME, prediction_time)][example_day]
    yta_model = fit_yet_to_arrive(inputs.inpatient_arrivals)
    yet_to_arrive = predict_yet_to_arrive(
        yta_model, inputs.params, example_day, prediction_time, prediction_window
    )

    return AnalysisResults(
        inputs=inputs,
        splits=splits,
        prediction_times=prediction_times,
        models=models,
        prob_dists=prob_dists,
        example_day=example_day,
        prediction_time=prediction_time,
        prediction_window=prediction_window,
        yet_to_arrive=yet_to_arrive,
        total_demand=combine_demand(in_ed["agg_predicted"], yet_to_arrive),
    )


def build_figures(results: AnalysisResults) -> dict[str, Figure]:
    return {
        "madcap.png": madcap_figure(results),
        "epudd.png": epudd_figure(results),
        "beds_in_ed.png": in_ed_figure(results),
        "beds_yet_to_arrive.png": yet_to_arrive_figure(results),
        "beds_total.png": total_demand_figure(results),
    }


def summarise_demand(results: AnalysisResults) -> pd.Series:
    proba = results.total_demand["agg_proba"]
    cdf = proba.cumsum()
    in_ed = in_ed_prediction(results)
    return pd.Series(
        {
            "expected beds": (proba.index * proba).sum(),
            "median": np.searchsorted(cdf, 0.5),
            "90% chance of needing at most": np.searchsorted(cdf, 0.9),
            "observed from patients in ED": in_ed["agg_observed"],
        }
    ).round(1)


