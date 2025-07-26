import os
import dotenv
dotenv.load_dotenv()

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import logging
import argparse
import json
import time

import methods, utils
from evaluator import build_evaluator


def generate(args):
    if args.few_shot:  # for few-shot, use the test set with 200 questions
        dataset = json.load(open("data_200_test_set.json"))
    else:
        dataset = json.load(open("Data_Merge.json"))
    assert args.method in ("cot", "cot_sc", "tot", "self-refine")  # currently only cot and cot_sc are supported

    cls = getattr(methods, args.method_id)
    method = cls(args)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(dataset)
    logging.info(f"=====Will evaluate {end_idx - start_idx} questions=====")
    results = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if args.few_shot:
        exp_dir = f"logs/exp_i"
        os.makedirs(exp_dir, exist_ok=True)
        file_prefix = f"results_{args.model}_{args.method}_fewshot_{args.shot_num}_{start_idx}_{end_idx}_{timestamp}"
    else:
        exp_dir = "logs"
        os.makedirs(exp_dir, exist_ok=True)
        file_prefix = f"results_{args.model}_{args.method}_{start_idx}_{end_idx}_{timestamp}"

    results_path = os.path.join(exp_dir, f"{file_prefix}.xlsx")
    result_path_json = os.path.join(exp_dir, f"{file_prefix}.json")
    for i in tqdm(range(start_idx, end_idx)):
        question, answer = dataset[i]['question'], dataset[i]['answer']
        rationales, prediction = method.run(question)

        result_dict = {
            "index": i,
            "QA_Type": dataset[i]['QA_Type'],
            "question": question,
            "answer": answer,
            "explanation": dataset[i]['explanation'],
            "Scoring_Points_raw": dataset[i]['Scoring_Points_raw'],
            "Scoring_Points_expert": dataset[i]['Scoring_Points_expert'],
            "predicted_rationale": rationales,
            "prediction": prediction
        }
        results.append(result_dict)
        if i % 5 == 0:
            with open(result_path_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    # write results to Excel file
    df_results = pd.DataFrame(results)  # create results DataFrame
    utils.beautify_write_excel(df_results, results_path)
    with open(result_path_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    summary = {
        "model": args.model,
        "start_idx": start_idx,
        "end_idx": end_idx
    }
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.few_shot:
        exp_dir = f"logs/exp_i"
        file_prefix = f"summary_{args.model}_{args.method}_fewshot_{args.shot_num}_{start_idx}_{end_idx}_{timestamp}"
    else:
        exp_dir = "logs"
        file_prefix = f"summary_{args.model}_{args.method}_{start_idx}_{end_idx}_{timestamp}"
    summary_path = os.path.join(exp_dir, f"{file_prefix}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)


def evaluate(evaluator, args):
    ## 1. Load results
    files = [f for f in os.listdir("logs") if f.startswith(f"results_{args.model}_{args.method}") and f.endswith(".json")]
    if not files:
        raise ValueError("No results files found.")
    
    latest = sorted(files)[-1]
    input_path = os.path.join("logs", latest)
    logging.info(f"Loading results from {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f) # # Load all data from the JSON file

    ## 2. Evaluate
    for result in tqdm(results):
        # scoring points: list of string, answer: string(ground truth), rationale: string(model prediction full text)
        question, scoring_points, answer = result['question'], result['Scoring_Points_expert'], result['answer']
        rationale, prediction = result['predicted_rationale'], result['prediction']
        ## Eval the rationale score
        eval_result = evaluator.evaluate(question, rationale, scoring_points)
        if eval_result is None:
            continue
        for e in eval_result:
            result[f"{e['metric_name']}_rationale_score"] = e['score']
            result[f"{e['metric_name']}_rationale_eval_details"] = e['details']

            if evaluator.judge_method == "llm-explain-with-reference":
                result[f"correct_rationale"] = e['correct_rationale']
                result['Correct_rationale_num'] = e['correct_rationale_num']
            if evaluator.judge_method == "llm-explain-without-reference": # for llm-explain-without-reference, the details is a list of dicts, len = 1
                result[f"number_of_recalled_steps"] = e['details'][0].get('number_of_recalled_steps', 0)
                result[f"total_number_of_required_steps"] = e['details'][0].get('total_number_of_required_steps', 0)
        ## Eval the answer score
        answer_score = evaluator.evaluate_answer(prediction, answer)
        if answer_score:
            result[f"answer_score"] = 1
        else:
            result[f"answer_score"] = 0
        result['Total_rationale_gnd'] = len(result['Scoring_Points_expert'])

    ## 3. Compute global level rationale score & answer score
    score_keys = {
        k for k in results[0].keys()
        if k.endswith("_rationale_score") or k.endswith("_answer_score")
    }

    global_avg = {
        k: sum(r.get(k, 0.0) for r in results) / len(results)
        for k in score_keys
    }

    ## 4. Save the results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    eval_result_path_xlsx = f"logs/eval_results_{args.model}_{args.method}_{args.judge_method}_{args.judge_model}_{timestamp}.xlsx"
    eval_result_path_json = f"logs/eval_results_{args.model}_{args.method}_{args.judge_method}_{args.judge_model}_{timestamp}.json"
    with open(eval_result_path_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    ## Make a clean version of the results for excel
    for result in results:
        keys_to_remove = [k for k in result if k.endswith("_rationale_eval_details")]

        for k in keys_to_remove:
            del result[k]

    df_results = pd.DataFrame(results)
    utils.beautify_write_excel(df_results, eval_result_path_xlsx)
    logging.info(f"Eval results written to {eval_result_path_xlsx}")

    ## 5. Save the summary
    summary = {
        'results_file': latest,
        **global_avg,
        'config': args.__dict__
    }
    
    sum_file = f"logs/eval_summary_{args.model}_{args.method}_{args.judge_method}_{args.judge_model}_{timestamp}.json"
    with open(sum_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    logging.info(f"Eval summary written to {sum_file}")

    return global_avg



def get_args():
    parser = argparse.ArgumentParser()
    # ===Arguments for model and method===
    parser.add_argument("--model", type=str, required=True,
                        choices=["gpt-4o-mini", "gpt-4o", "o1", "o3", "claude", "claude-sonnet-3.5", "gemini", "gemini-2.5-flash", \
                                "Llama-3.3-70B", "Llama-3.3-70B-quantized", "Qwen2.5-32B", "DeepSeek-v3", "Meditron-70B", \
                                "Qwen3-32B", "QwQ-32B", "DeepSeek-R1", "Baichuan-M1-14B", "HuatuoGPT-o1-70B", \
                                "ClinicalCamel-70B", "Mellama-70B", "PMC-LLaMA-13B", "MedGemma-27B"])
    parser.add_argument("--method", type=str, required=True,
                        choices=["io", "cot", "cot_sc", "tot", "self-refine", "rstar", "medagents", "mdagents", "aflow", "spo"])    
    
    # ===Arguments for OPEN-SOURCE models===
    parser.add_argument("--temperature", type=float, default=0, help="temperature")
    parser.add_argument("--seed", type=int, default=42, help="seed for vllm inference")
    parser.add_argument("--max_new_tokens", type=int, default=3072, help="maximum number of newly generated tokens")
    parser.add_argument("--tensor_parallel", type=int, default=2, help="number of GPUs to use for running open-source models only.")
    parser.add_argument("--quantization", type=str, default=None, help="quantization", choices=["awq", "awq-4bit", "awq-8bit", "fp16", "bf16", "int8"])
    parser.add_argument("--gpu_utilization", type=float, default=0.9, help="GPU utilization")

    # ===Arguments for CoT method===
    parser.add_argument("--few_shot", action="store_true", help="Use few-shot examples for CoT")
    parser.add_argument("--shot_num", type=int, default=3, help="Number of shots for CoT")

    # ===Arguments for CoT_SC method===
    parser.add_argument("--sc_cnt", type=int, default=None, help="Sampling count for CoT_SC")

    # ===Arguments for ToT method===
    parser.add_argument("--tot_k", type=int, default=5, help="Number of thoughts to generate at each step for ToT")
    
    # ===Arguments for Self-Refine method===
    parser.add_argument("--max_iterations_self_refine", type=int, default=3, help="Maximum number of iterations for self-refine")

    # ===Arguments for generation===
    parser.add_argument("--gen", action="store_true", help="Only generate results")
    parser.add_argument('--start_idx', type=int, default=0, help='Start example index')
    parser.add_argument('--end_idx', type=int, default=None, help='End example index')
    
    # ===Arguments for Rationale Evaluation===
    parser.add_argument("--eval", action="store_true", help="Only evaluate existing results written in the logs directory")
    parser.add_argument("--judge_method", type=str, default="text-metric", choices=["llm-explain-with-reference", "llm-explain-without-reference", "text-metric"], help="judge method")
    parser.add_argument("--judge_model", type=str, default="None", help="judge model")
    parser.add_argument("--judge_kwargs", type=str, default='{"temperature": 0.1, "max_tokens": 4096}', help="llm as judge, judge generation kwargs, e.g. {'temperature': 0.0}")
    parser.add_argument("--text_metrics", type=str, nargs="+", default=["bleu", "rougeL", "meteor", "bleurt", "bertscore"], help="text metric, e.g. bleu,meteor,rougeL,bleurt,bertscore")
    
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Program starts")
    args = get_args()
    logging.info("Command line arguments:")
    for param, value in vars(args).items():
        logging.info("Argument %s: %s", param, value)
    
    # Commercial LLMs taht are called through API
    commercial_llms = ("gpt-4o-mini", "gpt-4o", "o1", "o3", "claude-sonnet-3.5", "gemini", "gemini-2.5-flash", "DeepSeek-R1", "DeepSeek-v3")
    args.commercial_llms = commercial_llms
    model2id = {
        "Llama3":"meta-llama/Llama-3.1-8B-Instruct",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o": "gpt-4o",
        "o1": "o1",
        "o3": "o3",
        "claude-sonnet-3.5": "claude-3-5-sonnet-20240620",
        "gemini-2.5-flash": "gemini-2.5-flash-preview-05-20",
        "Llama-3.3-70B": "meta-llama/Llama-3.3-70B-Instruct",
        "Llama-3.3-70B-quantized": "RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8",
        "Qwen2.5-32B": "Qwen/Qwen2.5-32B-Instruct",
        "DeepSeek-v3": "deepseek-chat",
        "Meditron-70B": "epfl-llm/meditron-70b",
        "Mellama-70B": "mellama-70b",
        "Qwen3-32B": "Qwen/Qwen3-32B",
        "QwQ-32B": "Qwen/QwQ-32B",
        "DeepSeek-R1": "deepseek-reasoner",
        "Baichuan-M1-14B": "baichuan-inc/Baichuan-M1-14B-Instruct",
        "HuatuoGPT-o1-70B": "FreedomIntelligence/HuatuoGPT-o1-70B",
        "ClinicalCamel-70B": "wanglab/ClinicalCamel-70B",
        "PMC-LLaMA-13B": "axiong/PMC_LLaMA_13B",
        "MedGemma-27B": "google/medgemma-27b-text-it"
    }
    method2id = {
        "io": "IO",
        "cot": "CoT",
        "cot_sc": "CoT_SC",
        "tot": "ToT",
        "self-refine": "Self_Refine",
        "rstar": "rStar",
        "medagents": "MedAgents",
        "mdagents": "MDAgents",
        "aflow": "AFlow",
        "spo": "SPO"
    }
    args.model_id = model2id[args.model]
    if args.eval:
        args.judge_model_id = model2id.get(args.judge_model, args.judge_model)
    args.method_id = method2id[args.method]
    logging.info(f"MODEL: {args.model} ({args.model_id})")
    logging.info(f"METHOD: {args.method} ({args.method_id})")

    if args.gen:
        generate(args)
    
    if args.eval:
        evaluator = build_evaluator(
            judge_method=args.judge_method,
            text_metrics=args.text_metrics,
            model_name=args.judge_model,
            model_id=args.judge_model_id,
            args=args,
            generation_kwargs=json.loads(args.judge_kwargs)
        )
        global_avg = evaluate(evaluator, args)
        for k, v in global_avg.items():
            logging.info(f"{k}: {v:.3f}")
    logging.info("Program ends.")
