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


def test_metrics(models: dict[str, TrainedClassifier]) -> pd.DataFrame:
    return pd.DataFrame(
        {key: model.training_results.test_results for key, model in models.items()}
    ).T.round(3)


# Data


def load_inputs() -> Inputs:
    ed_visits = load_data(
        DATA_DIR,
        file_name="ed_visits.csv",
        index_column="snapshot_id",
        sort_columns=["visit_number", "snapshot_date", "prediction_time"],
        eval_columns=["prediction_time", "consultation_sequence", "final_sequence"],
    )
    ed_visits["snapshot_date"] = pd.to_datetime(ed_visits["snapshot_date"]).dt.date

    inpatient_arrivals = load_data(
        DATA_DIR, file_name="inpatient_arrivals.csv", index_column="arrival_datetime"
    )
    inpatient_arrivals.index = pd.to_datetime(inpatient_arrivals.index)

    return Inputs(ed_visits, inpatient_arrivals, load_config_file(CONFIG_PATH))


def split_visits(inputs: Inputs) -> Splits:
    params = inputs.params
    train, calibration, validation, test = create_temporal_splits(
        inputs.ed_visits,
        params["start_training_set"],
        params["start_validation_set"],
        params["start_test_set"],
        params["end_test_set"],
        col_name="snapshot_date",
        visit_col="visit_number",
        start_calibration=params["start_calibration_set"],
        verbose=False,
    )
    return Splits(train, calibration, validation, test)


# Patients already in the ED


def train_models(
    splits: Splits, prediction_times: list[PredictionTime]
) -> dict[str, TrainedClassifier]:
    return {
        get_model_key(MODEL_NAME, pt): train_classifier(
            train_visits=splits.train,
            valid_visits=splits.validation,
            test_visits=splits.test,
            calibration_visits=splits.calibration,
            prediction_time=pt,
            grid={"n_estimators": [30]},
            exclude_from_training_data=EXCLUDE_FROM_TRAINING,
            ordinal_mappings=ORDINAL_MAPPINGS,
            visit_col="visit_number",
            use_balanced_training=True,
            calibrate_probabilities=True,
            calibration_method="isotonic",
            evaluate_on_test=True,
        )
        for pt in prediction_times
    }


def predict_bed_counts(
    models: dict[str, TrainedClassifier],
    visits: pd.DataFrame,
    prediction_times: list[PredictionTime],
) -> ProbDists:
    """For each snapshot time and day, the distribution of how many patients in the ED will be admitted."""
    prob_dists: ProbDists = {}
    for pt in prediction_times:
        key = get_model_key(MODEL_NAME, pt)
        X, y = prepare_patient_snapshots(
            df=visits, prediction_time=pt, single_snapshot_per_visit=False, visit_col="visit_number"
        )
        groups = prepare_group_snapshot_dict(visits[visits.prediction_time == pt])
        prob_dists[key] = get_prob_dist(groups, X, y, models[key])
    return prob_dists


def in_ed_prediction(results: AnalysisResults) -> dict[str, Any]:
    key = get_model_key(MODEL_NAME, results.prediction_time)
    return results.prob_dists[key][results.example_day]


# Patients yet to arrive


def fit_yet_to_arrive(inpatient_arrivals: pd.DataFrame) -> ParametricIncomingAdmissionPredictor:
    train = inpatient_arrivals[inpatient_arrivals.training_validation_test == "train"]
    model = ParametricIncomingAdmissionPredictor()
    model.fit(
        train,
        yta_time_interval=timedelta(minutes=15),
        num_days=train.index.normalize().nunique(),
    )
    return model


def predict_yet_to_arrive(
    model: ParametricIncomingAdmissionPredictor,
    params: dict[str, Any],
    day: date,
    prediction_time: PredictionTime,
    prediction_window: timedelta,
) -> pd.DataFrame:
    """Beds needed by patients arriving after prediction_time, if the ED meets the target in config.yaml."""
    return model.predict(
        prediction_time=prediction_time,
        prediction_window=prediction_window,
        prediction_date=day,
        x1=params["x1"],
        y1=params["y1"],
        x2=params["x2"],
        y2=params["y2"],
        max_value=50,
    )["unfiltered"]


def combine_demand(in_ed: pd.DataFrame, yet_to_arrive: pd.DataFrame) -> pd.DataFrame:
    """Sum of two independent bed counts: the convolution of their distributions."""
    total = np.convolve(in_ed["agg_proba"].to_numpy(), yet_to_arrive["agg_proba"].to_numpy())
    return pd.DataFrame({"agg_proba": total})


# Figures


def madcap_figure(results: AnalysisResults) -> Figure:
    return plot_madcap(
        results.models,
        results.splits.test,
        exclude_from_training_data=EXCLUDE_FROM_TRAINING,
        return_figure=True,
    )


def epudd_figure(results: AnalysisResults) -> Figure:
    return plot_epudd(
        results.prediction_times,
        results.prob_dists,
        model_name=MODEL_NAME,
        suptitle="EPUDD plots: observed vs predicted bed counts by snapshot time",
        return_figure=True,
    )


def in_ed_figure(results: AnalysisResults) -> Figure:
    in_ed = in_ed_prediction(results)
    fig = plot_prob_dist(
        in_ed["agg_predicted"],
        f"Beds needed by patients in the ED at {window_label(results)[0]} on {results.example_day}",
        include_titles=True,
        return_figure=True,
    )
    ax = fig.gca()
    ax.axvline(in_ed["agg_observed"], color="red", linestyle="--", label="Observed")
    ax.legend()
    return fig


def yet_to_arrive_figure(results: AnalysisResults) -> Figure:
    start, end = window_label(results)
    return plot_prob_dist(
        results.yet_to_arrive,
        f"Beds needed by patients arriving {start}–{end} on {results.example_day}"
        f"\n(if the {results.inputs.params['x1']}-hour target is met)",
        include_titles=True,
        truncate_at_beds=30,
        return_figure=True,
    )


def total_demand_figure(results: AnalysisResults) -> Figure:
    start, end = window_label(results)
    return plot_prob_dist(
        results.total_demand,
        f"Total beds needed {start}–{end} on {results.example_day}",
        include_titles=True,
        truncate_at_beds=60,
        probability_levels=[0.9, 0.7],
        return_figure=True,
    )


def window_label(results: AnalysisResults) -> tuple[str, str]:
    hour, minute = results.prediction_time
    start = timedelta(hours=hour, minutes=minute)
    end = (start + results.prediction_window) % timedelta(days=1)
    return format_clock(start), format_clock(end)


def format_clock(t: timedelta) -> str:
    minutes = int(t.total_seconds()) // 60
    return f"{minutes // 60:02d}:{minutes % 60:02d}"
