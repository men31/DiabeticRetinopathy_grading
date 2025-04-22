from typing import List, Dict, Tuple, Iterable, Union
import os, sys
import pickle

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype


from dr_grading.utils import load_dataset_from_hdf5
from dr_grading.ML import (
    build_gbm_pipelines,
    build_sklearn_pipelines,
    train_evaluate_models,
)


def dummy_multiclass_dataset(fraction: float = 0.1):
    from sklearn.datasets import fetch_covtype

    data_dict = fetch_covtype(data_home="data", download_if_missing=True, as_frame=True)
    X = data_dict["data"]
    y = data_dict["target"] - 1

    y = y.sample(frac=fraction, random_state=42)
    X = X.iloc[y.index].reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y


def suggestion_folder_name(curr_name: str, count: int = 1) -> str:
    new_name = f"{curr_name}_{count}"
    if os.path.exists(new_name):
        count += 1
        return suggestion_folder_name(curr_name, count)
    else:
        return new_name


# Create the result directory if it doesn't exist
result_dir = "DR_grading_SMOTE_dummy"
result_dir = suggestion_folder_name(result_dir)
os.makedirs(result_dir, exist_ok=True)


def main():

    # Load X and y
    # data_dir = r"D:\Aj_Aof_Work\OCT_Disease\DATASET\APTOS2019_V6\Original"
    # X_train, y_train = load_dataset_from_hdf5(os.path.join(data_dir, "train.h5"))
    # X_val, y_val = load_dataset_from_hdf5(os.path.join(data_dir, "val.h5"))
    # X_test, y_test = load_dataset_from_hdf5(os.path.join(data_dir, "test.h5"))
    X, y = dummy_multiclass_dataset(fraction=0.04)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    report_str = " "
    report_lst = []
    report_keys = ["Model", "CV F1", "Test F1"]

    pipeline = build_sklearn_pipelines(random_state=42)
    # pipeline = build_gbm_pipelines(random_state=42, mode="multi")
    pipeline.update(build_gbm_pipelines(random_state=42, mode="multi"))
    results_dict, best_model_result = train_evaluate_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        pipeline=pipeline,
        save_model_dir=result_dir,
    )

    print("\n===== Summary =====")
    report_str += "\n===== Summary ====="
    for name, res in results_dict.items():
        print(f"\n{name}")
        print(f"  CV  F1 : {res.cv_f1:.3f}")
        print(f"  Test F1: {res.test_f1:.3f}")
        print(f"  Params : {res.best_params}")
        report_str += f"\n{name}"
        report_str += f"\n  CV  F1 : {res.cv_f1:.3f}"
        report_str += f"\n  Test F1: {res.test_f1:.3f}"
        report_str += f"\n  Params : {res.best_params} \n"
        report_str += "----------------" * 15
        report_lst.append([name, res.cv_f1, res.test_f1])

    print("\n===== Best Overall (by CV F1) =====")
    print(best_model_result.best_estimator)
    print("Optimal hyper‑parameters:", best_model_result.best_params)
    print("=================" * 15)

    report_str += "\n===== Best Overall (by CV F1) ====="
    report_str += f"\n{best_model_result.best_estimator}"
    report_str += f"\nOptimal hyper‑parameters: {best_model_result.best_params}\n"
    report_str += "=================" * 15

    # Save the variables result_dict and best_model_result to a pickle file
    with open(os.path.join(result_dir, f"model_results.pkl"), "wb") as file:
        pickle.dump(
            {"result_dict": results_dict, "best_model_result": best_model_result}, file
        )

    # Save the report string to a text file
    with open(os.path.join(result_dir, f"model_report.txt"), "w") as file:
        file.write(report_str)

    # Save the report as a CSV file
    report_df = pd.DataFrame(report_lst, columns=report_keys)
    report_df.to_csv(
        os.path.join(result_dir, "model_performance_report.csv"), index=False
    )


if __name__ == "__main__":
    main()
