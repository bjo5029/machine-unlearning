# run_all_conditions.py (버그 수정된 최종본)

import os
import time
import importlib
import argparse
import copy

def main(args):
    print(f"Loading configuration from: {args.config_module}.py")
    config_module = importlib.import_module(args.config_module)
    
    from run_experiment import run_experiment

    # --- 실행할 조건 조합 정의 ---
    definitions = ['class']
    granularities = ["stage", "batch", "sample"]
    
    score_methods = config_module.CONFIG.get("score_methods_to_run", ["memorization", "es", "c_proxy"])
    print(f"Target score methods for this run: {score_methods}")
    
    ordering_combos = [
        ("easy_first", "easy_first"),
        ("easy_first", "hard_first"),
        ("hard_first", "easy_first"),
        ("hard_first", "hard_first"),
    ]

    all_experiments = []
    
    base_exp = { "forget_set_definition": "class" }

    for score in score_methods:
        for granularity in granularities:
            for f_order in ["easy_first", "hard_first"]:
                exp = copy.deepcopy(base_exp)
                exp.update({
                    "forget_partitioning_method": score,
                    "unlearning_granularity": granularity,
                    "forget_partition_ordering": f_order,
                    "use_retain_ordering": False
                })
                if granularity in ['batch', 'sample'] and f_order == 'hard_first':
                    continue
                all_experiments.append(exp)
        
        for f_order, r_order in ordering_combos:
            exp = copy.deepcopy(base_exp)
            exp.update({
                "forget_partitioning_method": score,
                "retain_partitioning_method": score,
                "unlearning_granularity": "stage",
                "use_retain_ordering": True,
                "forget_partition_ordering": f_order,
                "retain_partition_ordering": r_order,
            })
            all_experiments.append(exp)

    total_conditions = len(all_experiments)
    print("="*60)
    print(f"Total experiments to run in this process: {total_conditions}")
    print("="*60)
    
    start_time = time.time()

    for i, exp_params in enumerate(all_experiments, 1):
        importlib.reload(config_module)
        run_config = config_module.CONFIG.copy()
        run_config.update(exp_params)
        
        # ▼▼▼▼▼ [수정] 바로 이 부분의 키 이름을 수정했습니다. ▼▼▼▼▼
        fname_parts = [
            exp_params['forget_set_definition'],
            exp_params['forget_partitioning_method'],  # 'score_method' -> 'forget_partitioning_method'
            exp_params['unlearning_granularity'],
            exp_params['forget_partition_ordering']
        ]
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        if exp_params.get('use_retain_ordering'):
            fname_parts.append(f"paired_{exp_params['retain_ordering']}")
        
        results_filename = f"results_{'_'.join(fname_parts)}.csv"
        run_config["results_csv_filename"] = results_filename
        
        print(f"\n--- [{i}/{total_conditions}] Running Experiment ---")
        for key, val in exp_params.items():
            print(f"  - {key}: {val}")
        
        try:
            run_experiment(run_config)
            print(f"\n--- [{i}/{total_conditions}] Experiment FINISHED ---")
        except Exception as e:
            print(f"\n--- [{i}/{total_conditions}] Experiment FAILED ---")
            import traceback
            traceback.print_exc()
        print("="*60)
            
    end_time = time.time()
    print(f"All experiments in this process completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_module', type=str, default='config', help='Configuration file to use (e.g., config_gpu0)')
    args = parser.parse_args()
    main(args)
    