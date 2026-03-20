import pandas as pd
import numpy as np
import random
import os
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
from itertools import product
from river import stats,  preprocessing,  tree, drift
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from eval_bml_imp import eval_oml_imp_horizon
from spotriver.evaluation.eval_bml import plot_bml_oml_horizon_metrics
import psutil

ALIAS = "KS"
MEC_BATCHES = [["S1"], ["S2"], ["S3"]]
MRS = ["30"]
P_NUM = 1
M_NUM = 120000000
N = 1

SPLIT = 0
HORIZON = 1
GRACE_PERIOD = 0
OBSERVED_PATIENTS = ['A36HR6Y']
EXCLUDED_PATIENTS = ['AJ7TSV9','AS2MVDL']
FEATURES = ['hour', 'minute'] 

SEED = 1
np.random.seed(SEED)
random.seed(SEED)

new_labels = {
    'mean': 'Mean',
    'tree-ad_dd_ADWIN': 'ADWIN',
    'tree-ad_dd_KSWIN': 'KSWIN',
    'tree-ad_dd_DummyDriftDetector': 'Dummy',
    'tree-ad_dd_PageHinkley': 'PH',
}

class MeanRegressor:
    def __init__(self):
        self.mean = stats.Mean()

    def learn_one(self,x, y):
        self.mean.update(y)
        return self

    def predict_one(self,x):
        return self.mean.get()
    
param_grid = {
    # "mean": {},
    # "tree-no-cd": {},
    # "tree-ad-base": {},
    # "tree-ad-ph": {
    #     # 'dw': [10000, 20000, 30000],
    #     'd': [0.005, 0.1, 0.2],
    #     't': [90, 95],
    # },
    "tree-ad-ks": {
        'a': [0.001, 0.005],
        'w': [30000, 50000],
        's': [5000, 10000]
    },
    # "tree-ad-adwin": {
    #     'a': [0.005, 0.1, 0.001],
    #     'w': [10000, 20000, 30000],
    #     's': [10000, 20000, 30000]
    # }

}
    
MODEL_FACTORY = {
    "mean": {
        "builder": lambda params: MeanRegressor()
    },
    "tree-no-cd": {
        "builder": lambda params: (
            preprocessing.StandardScaler() |
            tree.HoeffdingAdaptiveTreeRegressor(
                grace_period=params.get("gp", 5000),
                max_depth=params.get("md", None),
                drift_window_threshold=params.get("dw", 30000),
                drift_detector=drift.NoDrift(),
                seed=SEED,
            )
        )
    },
    "tree-ad-base": {
        "builder": lambda params: (
            preprocessing.StandardScaler() |
            tree.HoeffdingAdaptiveTreeRegressor(
                grace_period=params.get("gp", 5000),
                max_depth=params.get("md", None),
                drift_window_threshold=params.get("dw", 30000),

                seed=SEED,
            )
        )
    },
    "tree-ad-ph": {
        "builder": lambda params: (
            preprocessing.StandardScaler() |
            tree.HoeffdingAdaptiveTreeRegressor(
                grace_period=params.get("gp", 5000),
                max_depth=params.get("md", None),
                drift_window_threshold=params.get("dw", 30000),
                # drift_detector=params.get("dd", drift.ADWIN()),
                drift_detector=drift.PageHinkley(
                    delta=params.get("d", 0.005),
                    threshold=params.get("t", 50.0)),
                seed=SEED,
            )
        )
    },

    "tree-ad-ks": {
        "builder": lambda params: (
            preprocessing.StandardScaler() |
            tree.HoeffdingAdaptiveTreeRegressor(
                grace_period=params.get("gp", 5000),
                max_depth=params.get("md", None),
                drift_window_threshold=params.get("dw", 30000),
                # drift_detector=params.get("dd", drift.ADWIN()),
                drift_detector=drift.KSWIN(
                    alpha=params.get("a", 0.005),
                    window_size=params.get("w", 10000),
                    stat_size=params.get("s", 10000),
                ),
                seed=SEED,
            )
        )
    },

    "tree-ad-adwin": {
        "builder": lambda params: (
            preprocessing.StandardScaler() |
            tree.HoeffdingAdaptiveTreeRegressor(
                grace_period=params.get("gp", 5000),
                max_depth=params.get("md", None),
                drift_window_threshold=params.get("dw", 30000),
                # drift_detector=params.get("dd", drift.ADWIN()),
                drift_detector=drift.ADWIN(
                    delta=params.get("a", 0.005)
                ),
                seed=SEED,
            )
        )
    }
}

def build_models(param_grid, factory):
    MODELS = {}

    for family, grid in param_grid.items():

        if not grid:
            MODELS[family] = factory[family]["builder"]({})
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

            MODELS[name] = base_model

    return MODELS

MODELS = build_models(param_grid, MODEL_FACTORY)

def process_single_mr(mech, mr, i, pat, folder_path_m, folder_path_imputed):
    local_result = {mr: {f"{imp}": 0 for imp in MODELS.keys()}}
    local_result[mr].update({f"t_{imp}": 0 for imp in MODELS.keys()})
    local_result[mr].update({f"it_{imp}": 0 for imp in MODELS.keys()})
    local_result[mr].update({f"m_{imp}": 0 for imp in MODELS.keys()})
    local_result[mr].update({f"med_{imp}": 0 for imp in MODELS.keys()})

    try:
        files_hr = [
            f for f in os.listdir(folder_path_m)
            if f.endswith(f"_hr_{mech}_{i}_{mr}.csv")
        ]
        if not files_hr:
            print(f"⚠️ Nenhum arquivo encontrado para {mr} em {pat}")
            return local_result
        
        df = pd.read_csv(os.path.join(folder_path_m, files_hr[0]))
        first_valid_idx = df['heartrate'].first_valid_index()
        df = df.loc[first_valid_idx:].reset_index(drop=True)
        df = df.iloc[:M_NUM]

        df_imputed = df[['datetime', 'heartrate', 'target']][SPLIT:].copy()
        df_imputed.reset_index(drop=True, inplace=True)

        df['datetime'] = pd.to_datetime(df['datetime'])
        for feat in FEATURES:
            df[feat] = df['datetime'].dt.__getattribute__(feat)
        df.drop(columns=["datetime"], inplace=True)

        pat_evals = {}

        models_to_process = list(MODELS.items())

        with ProcessPoolExecutor() as executor:
            futures = {}
            for imp_name, model in models_to_process:
                futures[executor.submit(
                    eval_oml_imp_horizon,
                    model=model,
                    train=df.iloc[:SPLIT].copy(),
                    test=df.iloc[SPLIT:].copy(),
                    imp_column="heartrate",
                    target_column="target",
                    horizon=HORIZON,
                    include_remainder=True,
                    metric=root_mean_squared_error,
                    oml_grace_period=GRACE_PERIOD,
                )] = imp_name

            for future in as_completed(futures):
                imp_name = futures[future]
                evals_oml, df_true_oml = future.result()
                
                pat_evals[imp_name] = evals_oml

                local_result[mr][imp_name] += evals_oml['Metric'].mean()
                local_result[mr][f"med_{imp_name}"] += evals_oml['Metric'].median()
                local_result[mr][f"t_{imp_name}"] += evals_oml['CompTime (s)'].sum()
                local_result[mr][f"it_{imp_name}"] = evals_oml['CompTime (s)'].mean()
                local_result[mr][f"m_{imp_name}"] += evals_oml['Memory (MB)'].sum()

                path_imp = os.path.join(folder_path_imputed, f"{imp_name}")
                os.makedirs(path_imp, exist_ok=True)

                df_imputed.loc[df_true_oml["Prediction"].index, 'heartrate'] = df_true_oml["Prediction"].values

                df_imputed.to_csv(
                    os.path.join(
                        path_imp,
                        f"{pat.rstrip('/').split('/')[-1]}_hr_{mech}_{i}_{mr}_{imp_name}.csv"
                    ),
                    index=False
                )

    except Exception as e:
        print(f"❌ Erro ao processar {pat.rstrip('/').split('/')[-1]} | Mechanism: {mech} | MR: {mr} | Dataset: {i} | Error: {e}")

    return local_result, pat_evals


def process_mechanism(mech, num_datasets, mrs):
    local_results = {mech: {}}
    patient_results = {mech: {}}        


    folder_path_m_base = os.path.join(
        os.path.dirname(__file__), 
        f"../Data/COVID-19-Wearables-MMD/{mech}/"
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
        for pat in tqdm(patients, desc=f"Processing mechanism {mech}"):
    
            patient_id = pat.rstrip('/').split('/')[-1]
            if patient_id not in patient_results[mech]:
                patient_results[mech][patient_id] = {}

            folder_path_imputed = os.path.join(
                os.path.dirname(__file__), 
                f"../Data/COVID-19-Wearables-Input/{mech}/{patient_id}/{i}/"
            )
            os.makedirs(folder_path_imputed, exist_ok=True)

            folder_path_m = f"{pat}/{i}"

            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        process_single_mr, mech, mr, i, pat, folder_path_m, folder_path_imputed
                    ): mr for mr in mrs
                }

                for future in as_completed(futures):
                    mr = futures[future]
                    try:
                        result = future.result()

                        if patient_id == 'A0NVTRV':
                            if result[1]:
                                df_labels = list(result[1].keys())
                                df_labels = [new_labels.get(imp, imp) for imp in df_labels if imp in MODELS.keys()]
                                evals_list = [evals for evals in result[1].values()]
                                for ev in evals_list:
                                    ev.dropna(inplace=True)
                                    ev.reset_index(drop=True, inplace=True)

                                plot_bml_oml_horizon_metrics(evals_list, df_labels, metric=root_mean_squared_error, filename=f"Plots/oml_{patient_id}_{mech}_{ALIAS}.png")

                        if mr not in patient_results[mech][patient_id]:
                            patient_results[mech][patient_id][mr] = {}

                        for mr_key, mr_dict in result[0].items():
                            patient_results[mech][patient_id][mr_key] = mr_dict.copy()

                        for mr_key, mr_dict in result[0].items():
                            if mr_key not in local_results[mech]:
                                local_results[mech][mr_key] = mr_dict.copy()
                            else:
                                for k, v in mr_dict.items():
                                    local_results[mech][mr_key][k] = local_results[mech][mr_key].get(k, 0) + v


                        process = psutil.Process(os.getpid())
                        memoria = process.memory_info().rss  # em bytes
                        print(f"Uso de memória {patient_id}: {memoria / 1024**2:.2f} MB")
                    
                    except Exception as e:
                        print(f"❌ Falha no MR {mr}: {e}")

    for mr in local_results[mech]:
        for imp in MODELS.keys():
            local_results[mech][mr][imp] /= len(patients) * num_datasets
            local_results[mech][mr][f"med_{imp}"] /= len(patients) * num_datasets
            local_results[mech][mr][f"t_{imp}"] /= len(patients) * num_datasets
            local_results[mech][mr][f"it_{imp}"] /= len(patients) * num_datasets
            local_results[mech][mr][f"m_{imp}"] /= len(patients) * num_datasets

    

    return {
        "c": local_results,
        "p": patient_results
    }

if __name__ == "__main__":

    combined_results = {}
    combined_pat_results = {}

    print(f"🚀 Iniciando processamento para mecanismos: {MEC_BATCHES} | Missing Rates: {MRS} | Datasets por paciente: {N} | Pacientes: {P_NUM}")
    print(f"Modelos a serem avaliados {len(MODELS)}: {list(MODELS.keys())}")

    total_time_start = time.time()

    for mechanisms in MEC_BATCHES:
        time_start = time.time()
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(process_mechanism, mech, N, MRS): mech
                for mech in mechanisms
            }
            for future in as_completed(futures):
                mech_name = futures[future]
                try:
                    result = future.result()

                    combined = result["c"]
                    per_patient = result["p"]
                    combined_results.update(combined)

                    if mech_name not in combined_pat_results:
                        combined_pat_results[mech_name] = per_patient[mech_name]
                    else:
                        for pat in per_patient[mech_name]:
                            combined_pat_results[mech_name][pat] = per_patient[mech_name][pat]

                except Exception as e:
                     print(f"\n❌ Erro ao processar {mech_name}: {e}")

            time_elapsed = time.time() - time_start
            hours = time_elapsed / 3600
            print(f"\n⏱️ Tempo de execução para mecanismos {mechanisms}: {hours:.2f} horas")

            records = []
            for mech in combined_results:
                for mr in combined_results[mech]:
                    for imp in MODELS.keys():
                        records.append({
                            'mechanism': mech,
                            'missing_rate': mr,
                            'imputer': imp,
                            'mean_rmse': round(combined_results[mech][mr][imp], 2),
                            'med_rmse': round(combined_results[mech][mr][f"med_{imp}"], 2),
                            'ac_time': round(combined_results[mech][mr][f't_{imp}'], 2),
                            'it_time': round(combined_results[mech][mr][f'it_{imp}'], 4),
                            'memory': round(combined_results[mech][mr][f'm_{imp}'], 2)
                        })

            results_df = pd.DataFrame(records)
            result_path = f'Analysis/Parameters/imputation_results_{ALIAS}.csv'
            if os.path.exists(result_path):
                os.remove(result_path)
            results_df.to_csv(result_path, index=False)
            print(f"✅ Resultados consolidados salvos em {result_path}")

            pat_records = []
            for mech in combined_pat_results:
                for pat in combined_pat_results[mech]:
                    for mr in combined_pat_results[mech][pat]:
                        for imp in MODELS.keys():
                            if imp in combined_pat_results[mech][pat][mr]:
                                pat_records.append({
                                    'mechanism': mech,
                                    'patient': pat,
                                    'missing_rate': mr,
                                    'imputer': imp,
                                    'mean_rmse': round(combined_pat_results[mech][pat][mr][imp], 2),
                                    'med_rmse': round(combined_pat_results[mech][pat][mr][f"med_{imp}"], 2),
                                    'ac_time': round(combined_pat_results[mech][pat][mr][f"t_{imp}"], 2),
                                    'it_time': round(combined_pat_results[mech][pat][mr][f"it_{imp}"], 4),
                                    'memory': round(combined_pat_results[mech][pat][mr][f"m_{imp}"], 2)
                                })

            pat_df = pd.DataFrame(pat_records)
            pat_path = f"Analysis/Parameters/imputation_results_by_patient_{ALIAS}.csv"
            if os.path.exists(pat_path):
                os.remove(pat_path)
            pat_df.to_csv(pat_path, index=False)
            print(f"✅ Resultados por paciente salvos em {pat_path}")

    total_time_elapsed = time.time() - total_time_start
    hours = total_time_elapsed / 3600
    print(f"\n⏱️ Tempo total de execução: {hours:.2f} horas")






