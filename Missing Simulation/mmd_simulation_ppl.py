import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from mdatagen.univariate.uMCAR import uMCAR
from mdatagen.univariate.uMAR import uMAR
from mdatagen.univariate.uMNAR import uMNAR

MR_F = 30
# MR_N = 3
SCENARIOS = ['S1']
NUM_DATASETS= 1

SPLITS_RATIO_S1 = [0.34, 0.33, 0.33]
MECHS_S1 = ["MAR", "MCAR", "MNAR"]

SPLITS_RATIO_S2 = [0.25, 0.25, 0.25, 0.25]
MECHS_S2 = ["MAR", "MNAR", "MAR", "MNAR"]

SPLITS_RATIO_S3 = [0.2, 0.2, 0.2, 0.2, 0.2]
MECHS_S3 = ["MNAR", "MAR", "MNAR", "MAR", "MNAR"]

SEED = 1

def simulate_mm(mech, X_split, mr_f, mnar_t=1):

    mech = mech.split("_")[0] if "_" in mech else mech

    if mech == "MCAR":
        generator = uMCAR(
            X=X_split.set_index("datetime"),
            y=X_split.heartrate.to_numpy(),
            missing_rate=mr_f,
            x_miss="heartrate",
            seed=SEED
        )
        generate_data = generator.random().reset_index()

    elif mech == "MAR":
        X_split["time"] = pd.to_datetime(X_split["datetime"]).dt.time
        generator = uMAR(
            X=X_split,
            y=X_split.heartrate.to_numpy(),
            missing_rate=mr_f,
            x_miss='heartrate',
            x_obs='time',
            seed=SEED
        )
        generate_data = generator.lowest().reset_index()

    elif mech == "MAR-m":
        X_split.reset_index(drop=True, inplace=True)
        X_split["time"] = pd.to_datetime(X_split["datetime"]).dt.time
        X_split["time"] = pd.to_timedelta(X_split["time"].astype(str))
        generator = uMAR(
            X=X_split,
            y=X_split.heartrate.to_numpy(),
            missing_rate=mr_f,
            x_miss='heartrate',
            x_obs='time',
            seed=SEED
        )
        generate_data = generator.median().reset_index()
        generate_data["row_id"] = X_split["row_id"].to_numpy()

    elif mech == "MNAR":
        generator = uMNAR(
            X=X_split.reset_index(drop=True),
            y=X_split.heartrate.to_numpy(),
            threshold=mnar_t,
            missing_rate=mr_f,
            x_miss='heartrate',
            seed=SEED
        )
        generate_data = generator.run(deterministic = False).reset_index()

    else:
        raise ValueError(f"Mechanism {mech} not recognized")

    return generate_data


def process_file(file_name, scenario, folder_path_o,
                 mr_f, num_datasets):
    
    all_returns = []
    
    for i in range(1, num_datasets + 1):

        # if mr == 1:
        #     file_path = os.path.join(folder_path_o, file_name)
        # else:
        #     folder_path_m = os.path.join(
        #         os.path.dirname(__file__), '..', 'Data',
        #         'COVID-19-Wearables-MMD',
        #         scenario, file_name.split('_')[0], str(i)
        #     )
        #     file_path = os.path.join(
        #         folder_path_m,
        #         file_name.replace(
        #             '.csv',
        #             f'_{scenario}_{i}_{((mr-1)*mr_f):02d}.csv'
        #         )
        #     )

        file_path = os.path.join(folder_path_o, file_name)
            
        try:
            data = pd.read_csv(file_path)
            data["target"] = data["heartrate"].astype(float)
        except Exception as e:
            print(f"Error loading file {file_name}: {e}")
            continue

        data = data.reset_index(drop=True)
        data["row_id"] = data.index

        X = data[["datetime", "heartrate", "row_id"]]
        X = X[~X["heartrate"].isna()].copy()

        mmd = {}
        idx_list = []
        
        match scenario:
            case "S1":
                split_sizes = [int(len(X) * ratio) for ratio in SPLITS_RATIO_S1]
                indexes = np.cumsum([0] + split_sizes)
                idx_list = indexes

                mms = {
                    f"{mech}_{i}": X.iloc[indexes[i]:indexes[i+1]] for i, mech in enumerate(MECHS_S1)
                }
                split_mr = mr_f / len(SPLITS_RATIO_S1)
                
                for mech, X_split in mms.items():
                    mmd[mech] = simulate_mm(mech, X_split.copy(), split_mr)

            case "S2":
                split_sizes = [int(len(X) * ratio) for ratio in SPLITS_RATIO_S2]
                indexes = np.cumsum([0] + split_sizes)
                idx_list = indexes

                split_mr = mr_f / len(SPLITS_RATIO_S2)
                
                mms = {
                    f"{mech}_{i}": X.iloc[indexes[i]:indexes[i+1]] for i, mech in enumerate(MECHS_S2)
                }

                for mech, X_split in mms.items():
                    mmd[mech] = simulate_mm(mech, X_split.copy(), split_mr)

            case "S3":
                split_sizes = [int(len(X) * ratio) for ratio in SPLITS_RATIO_S3]
                indexes = np.cumsum([0] + split_sizes)
                idx_list = indexes
                
                mms = {
                    f"{mech}_{i}": X.iloc[indexes[i]:indexes[i+1]] for i, mech in enumerate(MECHS_S3)
                }
                

                split_mr = mr_f / len(SPLITS_RATIO_S3)

                for mech, X_split in mms.items():
                    mmd[mech] = simulate_mm(mech, X_split.copy(), split_mr)
                
            case _:
                continue

        # -------- MERGE --------
        for mech, generate_data in mmd.items():
            data.loc[
                data["row_id"].isin(generate_data["row_id"]),
                "heartrate"
            ] = generate_data["heartrate"].to_numpy()

        data.drop(columns="row_id", inplace=True)

        # -------- SAVE --------
        base_folder = os.path.join(
            os.path.dirname(__file__),
            '..', 'Data', 'COVID-19-Wearables-MMD'
        )

        iteration_folder = os.path.join(
            base_folder,
            scenario,
            file_name.split("_")[0],
            str(i)
        )

        os.makedirs(iteration_folder, exist_ok=True)

        save_path = os.path.join(
            iteration_folder,
            file_name.replace(
                '.csv',
                f'_{scenario}_{i}_{MR_F:02d}.csv'
            )
        )

        data = data[["datetime", "heartrate", "target"]]
        data.to_csv(save_path, index=False)

        all_returns.append((file_name.split("_")[0], i, idx_list))

    return all_returns

if __name__ == "__main__":


    folder_path_o = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'Datasets', 'COVID-19-Wearables'
    )

    files = [
        file for file in os.listdir(folder_path_o)
        if file.endswith('_hr.csv')
    ]

    for scn in SCENARIOS:

        print(f"\nProcessing mechanism {scn}")
        results = []

        with ProcessPoolExecutor() as executor:

            futures = [
                executor.submit(
                    process_file,
                    file_name,
                    scn,
                    folder_path_o,
                    MR_F,
                    NUM_DATASETS
                )
                for file_name in files
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Files processed"):
                result = future.result()

                if result is not None:
                    results.extend(result)   

        summary_data = []

        max_len = max(len(idx_list) for _, _, idx_list in results)

        for file_id, iteration, idx_list in results:
            row = {
                "file_id": file_id,
                "dataset_iteration": iteration,
                "scenario": scn
            }

            for i in range(max_len):
                row[f"index_{i}"] = idx_list[i] if i < len(idx_list) else None

            summary_data.append(row)

        df_summary = pd.DataFrame(summary_data)

        summary_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'Data', 'COVID-19-Wearables-MMD',
            f"summary_{scn}_{MR_F:02d}.csv"
        )

        df_summary.to_csv(summary_path, index=False)
