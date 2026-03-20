import numpy as np
import pandas as pd
from spotriver.evaluation.eval_bml import ResourceMonitor, evaluate_model, gen_sliding_window, gen_horizon_shifted_window
from river import stream as river_stream
from typing import Tuple
from tqdm import tqdm


def eval_bml_imp_horizon(
    model: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    imp_column: str,
    target_column: str,
    horizon: int,
    include_remainder: bool = True,
    metric: object = None,
) -> tuple:
    
    # Check if metric is None or null and raise ValueError if it is
    if metric is None:
        raise ValueError("The 'metric' parameter must not be None or null.")
    # Reset index of train and test dataframes
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    # Initialize lists for predictions and differences
    preds_list = []
    diffs_list = []
    # Fit the model on the training data
    rm = ResourceMonitor()
    with rm:
        try:
            model.fit(
                train.loc[:, ~train.columns.isin([target_column, imp_column])],
                train[target_column]
            )
        except Exception as e:
            print(f"Train data: {train}")
            print(f"An error occurred while fitting the model: {e}")
    # Evaluate the model on empty arrays to get initial resource usage
    df_eval = pd.DataFrame.from_dict(
        [evaluate_model(y_true=np.array([]), y_pred=np.array([]), memory=rm.memory, r_time=rm.r_time, metric=metric)]
    )
    # If include_remainder is False, remove remainder rows from test dataframe
    if include_remainder is False:
        remainder = len(test) % horizon
        if remainder > 0:
            test = test[:-remainder]
    # Evaluate the model on batches of size horizon from the test dataframe
    for batch_number, batch_df in tqdm(test.groupby(np.arange(len(test)) // horizon), desc="Evaluating batches"):

        nan_indexes = batch_df[batch_df[imp_column].isna()].index
        if len(nan_indexes) == 0:
            continue

        rm = ResourceMonitor()
        with rm:
            try:
                preds = model.predict(
                    batch_df.loc[
                        nan_indexes,
                        ~batch_df.columns.isin([target_column, imp_column])
                    ]
                )
                # preds = model.predict(batch_df.loc[:, batch_df.columns.notin([target_column, imp_column])])
            except Exception as e:
                print(f"Batch data: {batch_df}")
                print(f"An error occurred while predicting: {e}")

        diffs = batch_df.loc[nan_indexes, target_column].values - preds
        df_eval.loc[batch_number + 1] = pd.Series(
            evaluate_model(
                y_true=batch_df.loc[nan_indexes, target_column],
                y_pred=preds,
                memory=rm.memory,
                r_time=rm.r_time,
                metric=metric,
            )
        )
        # Append predictions and differences to their respective lists
        preds_list.append(preds)
        diffs_list.append(diffs)
    # Concatenate predictions and differences lists into series
    series_preds = pd.Series(np.concatenate(preds_list))
    series_diffs = pd.Series(np.concatenate(diffs_list))
    # Create a dataframe with true values and add columns for predictions and differences
    # df_true = pd.DataFrame(test[target_column])
    df_true = pd.DataFrame(test[target_column][test[imp_column].isna()]).reset_index(drop=True)
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true

def eval_bml_imp_landmark(
    model: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    imp_column: str,
    horizon: int,
    include_remainder: bool = True,
    metric: object = None,
) -> tuple:
    
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    series_preds = pd.Series(dtype=float)
    series_diffs = pd.Series(dtype=float)
    rm = ResourceMonitor()
    with rm:
        model.fit(
            train.loc[:, ~train.columns.isin([target_column, imp_column])],
            train[target_column]
        )
    df_eval = pd.DataFrame.from_dict(
        [evaluate_model(y_true=np.array([]), y_pred=np.array([]), memory=rm.memory, r_time=rm.r_time, metric=metric)]
    )
    if include_remainder is False:
        rem = len(test) % horizon
        if rem > 0:
            test = test[:-rem]
    # Landmark Evaluation
    for i, new_df in tqdm(enumerate(gen_sliding_window(test, horizon)), desc="Evaluating landmark", total=len(test)//horizon):
        new_df_not_nan = new_df[new_df[imp_column].notna()] 
        nan_indexes = new_df[new_df[imp_column].isna()].index
        if len(nan_indexes) == 0:
            continue
        train = pd.concat([train, new_df_not_nan], ignore_index=True)
        rm = ResourceMonitor()
        with rm:
            preds = pd.Series(model.predict(
                new_df.loc[
                    nan_indexes,
                    ~new_df.columns.isin([target_column, imp_column])
                ]
            ))
            # preds = pd.Series(model.predict(new_df.loc[:, new_df.columns != target_column]))
            model.fit(train.loc[:, ~train.columns.isin([target_column, imp_column])], train[target_column])
        diffs = new_df.loc[nan_indexes, target_column].values - preds
        df_eval.loc[i + 1] = pd.Series(
            evaluate_model(
                y_true=new_df.loc[nan_indexes, target_column],
                y_pred=preds,
                memory=rm.memory,
                r_time=rm.r_time,
                metric=metric,
            )
        )
        series_preds = pd.concat([series_preds, preds], ignore_index=True)
        series_diffs = pd.concat([series_diffs, diffs], ignore_index=True)
    df_true = pd.DataFrame(test[target_column][test[imp_column].isna()]).reset_index(drop=True)
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true

def eval_bml_imp_window(
    model: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    imp_column: str,
    target_column: str,
    horizon: int,
    include_remainder: bool = True,
    metric: object = None,
) -> tuple:
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    df_all = pd.concat([train, test], ignore_index=True)
    series_preds = pd.Series(dtype=float)
    series_diffs = pd.Series(dtype=float)
    rm = ResourceMonitor()
    with rm:
        model.fit(
            train.loc[:, ~train.columns.isin([target_column, imp_column])],
            train[target_column]
        )
    df_eval = pd.DataFrame.from_dict(
        [evaluate_model(y_true=np.array([]), y_pred=np.array([]), memory=rm.memory, r_time=rm.r_time, metric=metric)]
    )
    if include_remainder is False:
        rem = len(test) % horizon
        if rem > 0:
            test = test[:-rem]
    for i, (w_train, w_test) in tqdm(enumerate(gen_horizon_shifted_window(df_all, len(train), horizon)), desc="Evaluating windows", total=len(test)//horizon):
        rm = ResourceMonitor()
        nan_indexes = w_test[w_test[imp_column].isna()].index
        w_train_not_nan = w_train[w_train[imp_column].notna()]
        if len(nan_indexes) == 0:
            continue
        with rm:
            model.fit(
                w_train_not_nan.loc[:, ~w_train_not_nan.columns.isin([target_column, imp_column])],
                w_train_not_nan[target_column]
            )
            preds = pd.Series(model.predict(w_test.loc[nan_indexes, ~w_test.columns.isin([target_column, imp_column])]))
        diffs = w_test.loc[nan_indexes, target_column].values - preds
        df_eval.loc[i + 1] = pd.Series(
            evaluate_model(
                y_true=w_test.loc[nan_indexes, target_column],
                y_pred=preds,
                memory=rm.memory,
                r_time=rm.r_time,
                metric=metric,
            )
        )

        series_preds = pd.concat([series_preds, preds], ignore_index=True)
        series_diffs = pd.concat([series_diffs, diffs], ignore_index=True)

    df_true = pd.DataFrame(test[target_column][test[imp_column].isna()]).reset_index(drop=True)
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true

def eval_oml_imp_horizon(
    model: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    imp_column: str,
    target_column: str,
    horizon: int,
    include_remainder: bool = True,
    metric: object = None,
    oml_grace_period: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    # Check if metric is None or null and raise ValueError if it is
    if metric is None:
        raise ValueError("The 'metric' parameter must not be None or null.")
    if oml_grace_period is None:
        oml_grace_period = horizon
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    if include_remainder is False:
        rem = len(test) % horizon
        if rem > 0:
            test = test[:-rem]

    eval_data = []
    series_preds = []
    series_diffs = []


    # Fit the model on the train data, i.e., initial Training on Train Data.
    # This is performed on a limited subset only (oml_grace_period).
    # No predictions are made here, only the model is fitted.
    # Memory and runtime are measured for the model fitting
    if(not train.empty):
        train_X = train.loc[:, ~train.columns.isin([target_column, imp_column])]
        train_y = train[imp_column]
        train_X = train_X.tail(oml_grace_period)
        train_y = train_y.tail(oml_grace_period)
        rm = ResourceMonitor()
        with rm:
            try:
                # for xi, yi in tqdm(river_stream.iter_pandas(train_X, train_y), desc="Initial training on train data", total=len(train_X)):
                for xi, yi in river_stream.iter_pandas(train_X, train_y):
                    # Before v0.19 we had to call predict_one before learn_one
                    # in order for the whole pipeline to be updated.
                    # Since v0.19, calling learn_one in a pipeline will update each part
                    # of the pipeline in turn.
                    # Before v0.19, predict_one has to be called for updating the unsupervised parts
                    # of the pipeline.
                    # The following line, which returns y_pred, which is not used after v0.19:
                    # _ = model.predict_one(xi)
                    # model = model.learn_one(xi, yi)
                    # Starting with 0.21.0, the learn_one and learn_many methods of each estimator don't not
                    # return anything anymore.
                    # This is to emphasize that the estimators are stateful.
                    if ~np.isnan(yi):
                        model.learn_one(xi, yi)
            except Exception as e:
                print(f"train_X data: {train_X}")
                print(f"train_y data: {train_y}")
                print(f"An error occurred while fitting the model: {e}")

        # Create empty lists to collect data
        
        # Measure the costs of the initial training:
        # Add the evaluation of the model (memory and time, not predictions) on the train data to the eval_data list
        # A metric must not be passed to the evaluate_model function, because no predictions are made here
        # If a metric is passed, it will be ignored, because no predictions are passed to the evaluation function
        # So, metric=None and metric=mean_absolute_error will both work
        # Return res_dict = {"Metric": score, "Memory (MB)": memory, "CompTime (s)": r_time}
        eval_data.append(
            evaluate_model(y_true=np.array([]), y_pred=np.array([]), memory=rm.memory, r_time=rm.r_time, metric=metric)
        )

    # Test Data Evaluation
    # A sliding window of length horizon is used to evaluate the model on the test data
    # for i, new_df in tqdm(enumerate(gen_sliding_window(test, horizon)), desc="Evaluating data points", total=len(test)//horizon):
    for i, new_df in enumerate(gen_sliding_window(test, horizon)):
        preds = []
        nan_indexes = new_df[new_df[imp_column].isna()].index
        # if len(nan_indexes) == 0:
        #     continue
        test_X = new_df.loc[:, ~new_df.columns.isin([target_column, imp_column])]
        test_y = new_df[imp_column]
        rm = ResourceMonitor()
        with rm:
            try:
                for xi, yi in river_stream.iter_pandas(test_X, test_y):
                    if np.isnan(yi):
                        pred = model.predict_one(xi)
                        preds.append(round(pred,0))
                        # if pred > 300:
                        #     print(i, xi, pred)
                    else:
                        model.learn_one(xi, yi)
            except Exception as e:
                print(f"test_X data: {test_X}")
                print(f"test_y data: {test_y}")
                print(f"An error occurred while predicting: {e}")
        preds = pd.Series(preds)
        diffs = new_df.loc[nan_indexes, target_column].values - preds

        # Collect data in lists
        eval_data.append(
            evaluate_model(
                y_true=new_df.loc[nan_indexes, target_column].values, 
                y_pred=preds, 
                memory=rm.memory, 
                r_time=rm.r_time, 
                metric=metric
            )
        )
        series_preds.extend(preds)
        series_diffs.extend(diffs)

    # Create DataFrames from the collected data
    df_eval = pd.DataFrame(eval_data)
    
    df_true = pd.DataFrame(test[target_column][test[imp_column].isna()])
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true
    