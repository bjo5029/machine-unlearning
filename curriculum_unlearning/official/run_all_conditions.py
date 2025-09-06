# run_all_conditions.py (의존성 주입 방식 수정본)

import os
import time
import importlib
import argparse

def main(args):
    # 커맨드라인 인자로 받은 설정 파일을 동적으로 불러옵니다.
    print(f"Loading configuration from: {args.config_module}.py")
    config_module = importlib.import_module(args.config_module)
    
    # run_experiment 모듈에서 함수를 불러옵니다.
    from run_experiment import run_experiment

    # 실행할 조건을 정의합니다.
    definitions = [args.definition] 
    orderings = ["easy_first", "hard_first", "random"]
    
    total_conditions = len(definitions) * len(orderings)
    current_run = 0
    
    print("="*60)
    print(f"Starting sequential experiments for {total_conditions} conditions on GPU...")
    print(f"  - Target Definition Type: {args.definition}")
    print(f"  - Output Directory: {config_module.CONFIG['model_save_dir']}")
    print("="*60)
    
    start_time = time.time()

    for definition in definitions:
        for ordering in orderings:
            current_run += 1
            
            # 매번 설정 파일의 원본을 새로 불러와 사용합니다.
            importlib.reload(config_module)
            run_config = config_module.CONFIG.copy()
            
            run_config["forget_set_definition"] = definition
            run_config["forget_partition_ordering"] = ordering
            
            results_filename = f"results_{definition}_{ordering}.csv"
            run_config["results_csv_filename"] = results_filename
            
            print(f"\n--- [{current_run}/{total_conditions}] Running Experiment ---")
            print(f"  - Forget Set Definition: {run_config['forget_set_definition']}")
            print(f"  - Partition Ordering   : {run_config['forget_partition_ordering']}")
            print(f"  - Results will be saved to: {os.path.join(run_config['model_save_dir'], results_filename)}")
            
            try:
                # 수정된 run_experiment 함수에 설정 객체(run_config)를 직접 전달합니다.
                run_experiment(run_config)
                
                print(f"\n--- [{current_run}/{total_conditions}] Experiment FINISHED ---")
            except Exception as e:
                print(f"\n--- [{current_run}/{total_conditions}] Experiment FAILED ---")
                import traceback
                traceback.print_exc() # 더 자세한 에러 로그를 위해 추가

            print("="*60)
            
    end_time = time.time()
    print(f"All {total_conditions} experiments completed in {end_time - start_time:.2f} seconds.")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_module', type=str, required=True, help='Configuration file to use (e.g., config_gpu0)')
    parser.add_argument('--definition', type=str, required=True, choices=['class', 'random'], help='Forget set definition type to run')
    args = parser.parse_args()
    main(args)
    