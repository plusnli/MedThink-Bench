# 🌟 MedQA_Reasoning

## 📂 File Usage
- **main.py**: The main entry point of the program.
- **model_manager.py**: For unified model calling.
- **methods/**: Contains classes for baseline methods.
- **utils.py**: Utility functions to support various operations.

## ⚙️ Arguments in main.py
#### For model and method
- **model**: Specifies the model to be used.
- **method**: Defines the baseline method to apply.
#### For open-source LLMs inference with vLLM
- **temperature**: Sets the temperature for generation.
- **max_new_tokens**: Maximum number of tokens to generate.
- **tensor_parallel**: [For open-source LLMs only] Number of GPUs for inference. Default is 2.
- **quantization**: Specifies the quantization version for open-source LLMs (applicable when supported).
- **gpu_utilization**: Specifies the gpu memory utilization for initializing the vLLM model.
#### For CoT-SC
- **cnt**: Required only for the CoT-SC method. Indicating the sampling counts.
#### For ToT
- **tot_k**: Required only for ToT method. Indicating the sampling counts.
#### For generation
- **start_idx**: Starting index of the dataset.
- **end_idx**: Ending index of the dataset. Default is `None`, indicating the last data point.
- **gen**: Set and specify for generation.
#### For Rationale Evaluation
- **eval**: If set, only evaluate existing results written in the logs directory.
- **judge_method**: Specifies the judge method for rationale evaluation. Choices: "llm-explain-with-reference", "llm-explain-without-reference", "text-metric". Default is "llm-explain-with-reference".
- **judge_model**: Specifies the model used for judging. Default is "gpt-4o-mini".
- **judge_kwargs**: JSON string for judge generation kwargs, e.g. '{"temperature": 0.0}'.
- **text_metrics**: List of text metrics to use for evaluation, e.g. ["bleu", "rougeL", "meteor", "bleurt", "bertscore"].


## 📝 Evaluation

### Run Evaluation

#### 🔹 Text-Metrics Evaluation

Supports: `bleu`, `rougeL`, `meteor`, `bertscore`, `bleurt`

```bash
python main.py --model QwQ-32B --method cot --eval \
  --judge_method 'text-metric' \
  --judge_model gpt-4o-mini --judge_kwargs '{"temperature":0.0}'
```

#### 🔹 LLM-as-Judge (with reference)
> Uses both the gold rationale and the model rationale to guide judgment.

```bash
python main.py --model QwQ-32B --method cot --eval \
  --judge_method llm-explain-with-reference \
  --judge_model gpt-4o-mini --judge_kwargs '{"temperature":0.1, "max_tokens": 4096}'
```

#### 🔹 LLM-as-Judge (without reference)
> Judges based solely on the model’s output without access to the reference.

```bash
python main.py --model QwQ-32B --method cot --eval \
  --judge_method llm-explain-without-reference \
  --judge_model gpt-4o-mini --judge_kwargs '{"temperature":0.1, "max_tokens": 4096}'
```
> Note: We use a low temperature (0.1) to ensure deterministic judgment behavior.
### Output

Evaluation results will be saved in the `logs/` directory as:

* `eval_result_*.json` or `.xlsx` – detailed results
* `eval_summary_*.json` – metric summary
