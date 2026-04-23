import pandas as pd
import numpy as np
import random
import os
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
from itertools import product
from river import drift
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from spotriver.evaluation.eval_bml import plot_bml_oml_horizon_metrics
import matplotlib.pyplot as plt
import copy


ALIAS = "BASE"
# MEC_BATCHES = [["S1"],["S2"],["S3"]]
SCE_BATCHES = [["S1","S2","S3"]]
MRS = ["30"]
P_NUM = 1
M_NUM = 100000000
N = 1

SPLIT = 0
HORIZON = 1
GRACE_PERIOD = 0
OBSERVED_PATIENTS = ['A36HR6Y']
EXCLUDED_PATIENTS = ['AJ7TSV9','AS2MVDL'] #AUY8KYW muito grande
FEATURES = ['hour', 'minute'] 

SEED = 1
np.random.seed(SEED)
random.seed(SEED)


    
param_grid = {
    "ADWIN":{
        # 'd': [0.0002],
        'c': [10000]
    },
    # "PH": {},
    # 'KSWIN': {
    #     # 'a': [0.0001],
    #     # 'w': [5000],
    #     # 's': [500]
    # },
}
    
MODEL_FACTORY = {
    "ADWIN": {
        "builder": lambda params: drift.ADWIN(
            delta=params.get('d', 0.002),
            clock= params.get('c', 32),
        ),
    },
    "PH": {
        "builder": lambda params: drift.PageHinkley(),
    },
    'KSWIN': {
        "builder": lambda params: drift.KSWIN(
            alpha=params.get('a', 0.0001),
            window_size=params.get('w', 5000), 
            stat_size=params.get('s', 500)
        ),
    },
}

def build_models(param_grid, factory):
    DD = {}

    for family, grid in param_grid.items():

        if not grid:
            DD[family] = factory[family]["builder"]({})
            continue

        keys = list(grid.keys())
        values = list(grid.values())

        for combo in product(*values):
            params = dict(zip(keys, combo))

            base_model = factory[family]["builder"](params)

            name = family + "_" + "-".join(
                f"{k}_{str(v).replace('.', '')}" if k != 'arq' 
                else f"{k}_{str(v[0]).replace('.', '')}"
                for k, v in params.items()
            )

            DD[name] = base_model

    return DD

DD = build_models(param_grid, MODEL_FACTORY)

def drift_detection(dd, stream):
    dd_f = type(dd)(**dd._get_params())
    dd_m = type(dd)(**dd._get_params())

    drifts_full = []
    drifts_miss = []
    for i, val in enumerate(stream):
        if np.isnan(val[0]):
            dd_m.update(val[1])
            if dd_m.drift_detected:
                drifts_miss.append(i)
        dd_f.update(val[0])   
        if dd_f.drift_detected:
            drifts_full.append(i)
    
    return drifts_full, drifts_miss

def plot_data_drift(df, drifts_full, drifts_miss, row_summary, sce, pat, model_name):

    y = df.values

    fig = plt.figure(figsize=(7,3), tight_layout=True)
    ax1 = plt.subplot(111)
    ax1.grid()

    ax1.plot(y, marker='o', linewidth=.001, markersize=0.1, color='grey')

    splits = [c for c in row_summary.columns if c.startswith("index_")]
    splits = sorted(splits, key=lambda x: int(x.split("_")[1]))
    splits = splits[1:-1]

    for i, split in enumerate(splits):
        split_idx = row_summary[split].values[0]
        ax1.axvline(split_idx, color='green', linestyle='--',
                    label='Split' if i == 0 else "")

    if drifts_full is not None:
        for i, drift_detected in enumerate(drifts_full):
            ax1.axvline(drift_detected, color='red',
                        label='Drift (Full)' if i == 0 else "")

    if drifts_miss is not None:
        for i, drift_detected in enumerate(drifts_miss):
            ax1.axvline(drift_detected, color='blue',
                        label='Drift (Missing)' if i == 0 else "")

    ax1.legend(loc='upper right', fontsize=6)
    plt.savefig(f"Plots/dp_full_{sce}_{pat}_{model_name}.png", dpi=300)

def plot_miss_drift(df, drifts_full, drifts_miss, row_summary, sce, pat, model_name):

    miss_index = df[df['heartrate'].isna()].index
    y = df['target'].values

    fig = plt.figure(figsize=(7,3), tight_layout=True)
    ax1 = plt.subplot(111)
    ax1.grid()

    ax1.plot(y, marker='o', linewidth=0, markersize=0.5, color='grey')

    ax1.scatter(miss_index, y[miss_index],
            color='red', s=2, zorder=3, label='Missing')

    splits = [c for c in row_summary.columns if c.startswith("index_")]
    splits = sorted(splits, key=lambda x: int(x.split("_")[1]))
    splits = splits[1:-1]

    for i, split in enumerate(splits):
        split_idx = row_summary[split].values[0]
        ax1.axvline(split_idx, color='green', linestyle='--',
                    label='Split' if i == 0 else "")

    if drifts_miss is not None:
        for i, drift_detected in enumerate(drifts_miss):
            ax1.axvline(drift_detected, color='blue',
                        label='Drift (Missing)' if i == 0 else "")

    ax1.legend(loc='upper right', fontsize=6)
    plt.savefig(f"Plots/dp_miss_{sce}_{pat}_{model_name}.png", dpi=300)



def process_single_mr(sce, mr, i, pat, folder_path_m):

    try:
        files_hr = [
            f for f in os.listdir(folder_path_m)
            if f.endswith(f"_hr_{sce}_{i}_{mr}.csv")
        ]
        if not files_hr:
            print(f"⚠️ Nenhum arquivo encontrado para {mr} em {pat}")
            return
        
        df = pd.read_csv(os.path.join(folder_path_m, files_hr[0]))
        first_valid_idx = df['heartrate'].first_valid_index()
        df = df.loc[first_valid_idx:].reset_index(drop=True)
        df = df.iloc[:M_NUM]

        df['datetime'] = pd.to_datetime(df['datetime'])
        for feat in FEATURES:
            df[feat] = df['datetime'].dt.__getattribute__(feat)
        df.drop(columns=["datetime"], inplace=True)

        models_to_process = list(DD.items())

        df_sum = pd.read_csv(
            os.path.join('..', 'Data', 'COVID-19-Wearables-MMD',
                            f'summary_{sce}_{30}.csv')
        )

        row_summary = df_sum[
            (df_sum["file_id"] == pat.rstrip('/').split('/')[-1]) &
            (df_sum["dataset_iteration"] == 1)
        ]

        with ProcessPoolExecutor() as executor:
            futures = {}
            for imp_name, model in models_to_process:
                futures[executor.submit(drift_detection, model, df[['heartrate','target']].values)] = imp_name

            for future in as_completed(futures):
                imp_name = futures[future]
                try:
                    drifts_full, drifts_miss = future.result()
                    if drifts_full or drifts_miss:
                        # plot_data_drift(df['target'], drifts_full, drifts_miss, row_summary, sce, pat.rstrip('/').split('/')[-1], imp_name)
                        plot_miss_drift(df[['heartrate','target']], drifts_full, drifts_miss, row_summary, sce, pat.rstrip('/').split('/')[-1], imp_name)
                except Exception as e:
                    print(f"❌ Erro ao processar modelo {imp_name} para {pat} | Mechanism: {sce} | MR: {mr} | Dataset: {i} | Error: {e}")

    except Exception as e:
        print(f"❌ Erro ao processar {pat.rstrip('/').split('/')[-1]} | Mechanism: {sce} | MR: {mr} | Dataset: {i} | Error: {e}")

def process_mechanism(sce, num_datasets, mrs):

    folder_path_m_base = os.path.join(
        os.path.dirname(__file__), 
        f"../Data/COVID-19-Wearables-MMD/{sce}/"
    )

    patients = [
        os.path.join(folder_path_m_base, name)
        for name in os.listdir(folder_path_m_base)
        if os.path.isdir(os.path.join(folder_path_m_base, name))
    ]
    patients = [p for p in patients if p.rstrip('/').split('/')[-1] not in EXCLUDED_PATIENTS]
    
    patients = patients[:P_NUM]  
    # patients = [p for p in patients if p.rstrip('/').split('/')[-1] in OBSERVED_PATIENTS]

    for i in range(1, num_datasets + 1):
        for pat in tqdm(patients, desc=f"Processing mechanism {sce}"):

            folder_path_m = f"{pat}/{i}"

            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        process_single_mr, sce, mr, i, pat, folder_path_m
                    ): mr for mr in mrs
                }

                for future in as_completed(futures):
                    mr = futures[future]
                    try:
                        result = future.result()
                    
                    except Exception as e:
                        print(f"❌ Falha no MR {mr}: {e}")


if __name__ == "__main__":

    combined_results = {}
    combined_pat_results = {}

    print(f"🚀 Iniciando processamento para mecanismos: {SCE_BATCHES} | Missing Rates: {MRS} | Datasets por paciente: {N} | Pacientes: {P_NUM}")
    print(f"Modelos a serem avaliados {len(DD)}: {list(DD.keys())}")

    total_time_start = time.time()

    for scenarios in SCE_BATCHES:
        time_start = time.time()
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(process_mechanism, sce, N, MRS): sce
                for sce in scenarios
            }
            # for future in as_completed(futures):
            #     mech_name = futures[future]
            #     try:
            #         result = future.result()

            #         combined = result["c"]
            #         per_patient = result["p"]
            #         combined_results.update(combined)

            #         if mech_name not in combined_pat_results:
            #             combined_pat_results[mech_name] = per_patient[mech_name]
            #         else:
            #             for pat in per_patient[mech_name]:
            #                 combined_pat_results[mech_name][pat] = per_patient[mech_name][pat]

            #     except Exception as e:
            #          print(f"\n❌ Erro ao processar {mech_name}: {e}")

            time_elapsed = time.time() - time_start
            hours = time_elapsed / 3600
            print(f"\n⏱️ Tempo de execução para mecanismos {scenarios}: {hours:.2f} horas")

            # records = []
            # for mech in combined_results:
            #     for mr in combined_results[mech]:
            #         for imp in MODELS.keys():
            #             records.append({
            #                 'mechanism': mech,
            #                 'missing_rate': mr,
            #                 'imputer': imp,
            #                 'mean_rmse': round(combined_results[mech][mr][imp], 2),
            #                 'med_rmse': round(combined_results[mech][mr][f"med_{imp}"], 2),
            #                 'ac_time': round(combined_results[mech][mr][f't_{imp}'], 2),
            #                 'it_time': round(combined_results[mech][mr][f'it_{imp}'], 4),
            #                 'memory': round(combined_results[mech][mr][f'm_{imp}'], 2)
            #             })

            # os.makedirs(f'Analysis/Parameters/{ALIAS}', exist_ok=True)

            # results_df = pd.DataFrame(records)
            # result_path = f'Parameters/{ALIAS}/imputation_results_{ALIAS}_{P_NUM}.csv'
            # file_exists = os.path.exists(result_path)
            # results_df.to_csv(result_path, index=False, mode='a', header = not file_exists)
            # print(f"✅ Resultados consolidados salvos em {result_path}")

            # # result_path = f'Analysis/imputation_results_{ALIAS}.csv'
            # # if os.path.exists(result_path):
            # #     os.remove(result_path)
            # # results_df.to_csv(result_path, index=False)
            # # print(f"✅ Resultados consolidados salvos em {result_path}")

            # pat_records = []
            # for mech in combined_pat_results:
            #     for pat in combined_pat_results[mech]:
            #         for mr in combined_pat_results[mech][pat]:
            #             for imp in MODELS.keys():
            #                 if imp in combined_pat_results[mech][pat][mr]:
            #                     pat_records.append({
            #                         'mechanism': mech,
            #                         'patient': pat,
            #                         'missing_rate': mr,
            #                         'imputer': imp,
            #                         'mean_rmse': round(combined_pat_results[mech][pat][mr][imp], 2),
            #                         'med_rmse': round(combined_pat_results[mech][pat][mr][f"med_{imp}"], 2),
            #                         'ac_time': round(combined_pat_results[mech][pat][mr][f"t_{imp}"], 2),
            #                         'it_time': round(combined_pat_results[mech][pat][mr][f"it_{imp}"], 4),
            #                         'memory': round(combined_pat_results[mech][pat][mr][f"m_{imp}"], 2)
            #                     })

            # pat_df = pd.DataFrame(pat_records)

            # result_path = f'Parameters/{ALIAS}/imputation_results_by_patient_{ALIAS}_{P_NUM}.csv'
            # file_exists = os.path.exists(result_path)
            # pat_df.to_csv(result_path, index=False, mode='a', header = not file_exists)
            # print(f"✅ Resultados consolidados salvos em {result_path}")

            # # pat_path = f"Analysis/imputation_results_by_patient_{ALIAS}.csv"
            # # if os.path.exists(pat_path):
            # #     os.remove(pat_path)
            # # pat_df.to_csv(pat_path, index=False)
            # # print(f"✅ Resultados por paciente salvos em {pat_path}")

    total_time_elapsed = time.time() - total_time_start
    hours = total_time_elapsed / 3600
    print(f"\n⏱️ Tempo total de execução: {hours:.2f} horas")






