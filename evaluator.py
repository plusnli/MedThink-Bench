# ─────────────── evaluator.py ───────────────
'''
pip install evaluate bert-score
pip install bleurt-pytorch
pip install moverscore==2.0.2
python -m nltk.downloader punkt wordnet omw-1.4
absl-py rouge-score nltk
setuptools ..
numpy version ..
pyemd
'''

import os, json, logging, time, re, ast
from typing import Any
from abc import ABC, abstractmethod
from openai import OpenAI
import torch
from vllm import LLM, SamplingParams
from model_manager import ModelManager

import evaluate as hf_eval
from bert_score import score as bert_score_fn
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
# from moverscore import get_idf_dict, word_mover_score
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

_METRIC_LOADERS = {
    "bleu":    lambda: hf_eval.load("bleu"),
    "meteor":  lambda: hf_eval.load("meteor"),
    "rougeL":  lambda: hf_eval.load("rouge"),
}

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",      # ```json ... ```
    re.DOTALL | re.IGNORECASE
)

def extract_json_from_response(text: str) -> str:
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1)

    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return text.strip()

def safe_json_loads(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        _patched = raw.replace("'", '"')
        try:
            return json.loads(_patched)
        except json.JSONDecodeError:
            return ast.literal_eval(raw)

class BaseEvaluator(ABC):
    def __init__(self, judge_method: str):
        self.judge_method = judge_method
    @abstractmethod
    def evaluate(self, question: str, prediction: str, answer: str) -> tuple[bool, dict]:
        """
        return format:
        [
            {
                "metric_name": str,
                "score": float,
                "detail": list[dict]
            }
            ...
        ]
        """
    def evaluate_answer(self, prediction: str, answer: str) -> tuple[bool, dict]:
        if not prediction:
            return False
        answer = answer.upper()
        prediction = prediction.upper()
        if answer.startswith(prediction):
            return True
        else:
            return False

class TextMetricEvaluator(BaseEvaluator):
    def __init__(self, metrics: list[str] | None = None):
        super().__init__(judge_method="text-metric")
        self.metrics = metrics
        if metrics is None:
            self.metrics = ["rougeL", "bleu", "meteor", "bleurt", "bertscore"]
        if "bleurt" in self.metrics:
            config = BleurtConfig.from_pretrained("lucadiliello/BLEURT-20")
            self._bleurt_model = BleurtForSequenceClassification.from_pretrained("lucadiliello/BLEURT-20", config=config)
            self._bleurt_tokenizer = BleurtTokenizer.from_pretrained("lucadiliello/BLEURT-20")
            self._bleurt_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._bleurt_model.to(self._bleurt_device).eval()
        self.loaders = {}
        for metric in self.metrics:
            if metric in _METRIC_LOADERS:
                self.loaders[metric] = _METRIC_LOADERS[metric]()


    def evaluate(self, question, prediction: str, key_points: list[str], **_) -> tuple[bool, dict]:
        ref_text = "".join(key_points)
        if not prediction:
            return []

        scores = []
        for metric in self.metrics:
            if metric in _METRIC_LOADERS:
                if metric == "bleu":
                    score = self.loaders[metric].compute(predictions=[prediction], references=[[ref_text]])["bleu"]
                else:
                    score = self.loaders[metric].compute(predictions=[prediction], references=[ref_text])[metric]
                scores.append({
                    "metric_name": metric,
                    "score": score,
                    "details": []
                })
            
            elif metric == "bleurt":
                inputs = self._bleurt_tokenizer(
                [ref_text], [prediction],
                padding="longest", return_tensors="pt", truncation=True, max_length=512,
                ).to(self._bleurt_device)
                with torch.no_grad():
                    logits = self._bleurt_model(**inputs).logits.flatten()
                bleurt_scores = logits.cpu().tolist()
                scores.append({
                    "metric_name": metric,
                    "score": max(bleurt_scores),
                    "details": []
                })
            elif metric == "bertscore":
                bert_score_scores = bert_score_fn([prediction], [ref_text], lang="en", model_type="bert-base-uncased")
                scores.append({
                    "metric_name": metric,
                    "score": bert_score_scores[2].mean().item(),
                    "details": []
                })
            # elif metric == "moverscore":
            #     flatten_refs = [r for group in [ref_text] for r in ([group] if isinstance(group, str) else group)]
            #     idf_dict_hyp = get_idf_dict([prediction])
            #     idf_dict_ref = get_idf_dict(flatten_refs)
            #     moverscore_scores = word_mover_score(flatten_refs, [prediction],
            #                         idf_dict_ref, idf_dict_hyp,
            #                         stop_words=[], n_gram=1, remove_subwords=True, batch_size=8)
            #     scores.append({
            #         "metric_name": metric,
            #         "score": max(moverscore_scores),
            #         "details": []
            #     })
            else:
                raise ValueError(f"Unknown metric {metric}")
        logging.info("*"*100)
        logging.info(f"scores: {scores}")
        logging.info("*"*100)
        return scores

# -------- LLM-as-a-Judge --------

SYSTEM_PROMPT = """You are a meticulous medical QA grader.

Given a question, a ground-truth rationale, and a model’s prediction (which may include reasoning),
evaluate whether the prediction comprehensively mentions the ground-truth rationale.

Grade in a point-wise manner:
- The ground-truth rationale is considered mentioned only if **its main content** is **explicitly stated** in the model's prediction.
- **Do not award the statement** if any key information of the ground-truth rationale is missing from the model's prediction.
- **Do not award the statement** if any information in the model's prediction conflicts with the ground-truth rationale.
- **Do not award the statement** if only a small part of the ground-truth rationale is mentioned in the model's prediction.

Respond in the following JSON format:

{
  "judge_reason": "<brief explanation identifying why the prediction contains or does not contain the ground-truth rationale>",
  "contains_key_point": <contains_ground_truth_rationale> (a boolean value)
}
"""

SYSTEM_PROMPT_WITHOUT_KEY_POINT = """You are a meticulous medical QA grader.

Given a question and its answer, you should first generate a list of nuanced reasoning steps that can accurately lead to the correct answer. 
Then, you should take the generated reasoning steps as ground-truth rationale to assess the quality of model-generated rationales from a prediction model.


Specifically, given the question and a model’s prediction (which may include reasoning), evaluate whether the prediction comprehensively mentions your generated rationale (i.e., ground-truth rationale).
Grade in a point-wise manner:
- The ground-truth rationale is considered mentioned only if **its main content** is **explicitly stated** in the model's prediction.
- **Do not award the statement** if any key information of the ground-truth rationale is missing from the model's prediction.
- **Do not award the statement** if any information in the model's prediction conflicts with the ground-truth rationale.
- **Do not award the statement** if only a small part of the ground-truth rationale is mentioned in the model's prediction.

Return the following:
* `gold_steps`: (list the gold-standard reasoning steps)
* `recalled_steps`: (list the matched reasoning steps)
* `number_of_recalled_steps`: (integer)
* `total_number_of_required_steps`: (integer)

**Format your output exactly like this:**
```json
{
   "gold_steps": ["Step 1...", "Step 2...", "Step 3..."],
   "recalled_steps": ["Step 1...", "Step 3..."],
   "number_of_recalled_steps": 2,
   "total_number_of_required_steps": 3
}
```
"""




GRADE_TEMPLATE = """
## Please evaluate the model's prediction and provide your reasoning.

Question:
{question}

Scoring Point:
{scoring_point}

Model Prediction:
{prediction}
"""

GRADE_TEMPLATE_WITHOUT_KEY_POINT = """
## Please evaluate the model's prediction and provide your reasoning.

Question:
{question}

Model Prediction:
{prediction}
"""

class LLMJudge(BaseEvaluator):
    def __init__(self,
                 judge_method: str = "llm-explain-with-reference",
                 model_name: str = "gpt-4o-mini",
                 model_id: str = None,
                 reference: bool = True,
                 generation_kwargs: dict = {},
                 args: dict = {},
                 ):
        super().__init__(judge_method=judge_method)
        # self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.args = args

        self.client = ModelManager(args, model_name=model_name, model_id=model_id)
        self.client.sampling_params = SamplingParams(
            seed=args.seed,
            **generation_kwargs
        )
        self.model_name = model_name
        self.generation_kwargs = generation_kwargs
        if self.judge_method == "llm-explain-with-reference":
            self.reference = reference
        elif self.judge_method == "llm-explain-without-reference":
            self.reference = False
        else:
            raise ValueError(f"Unknown judge method: {self.judge_method}")
        if self.reference:
            self.system_prompt = SYSTEM_PROMPT
            self.template = GRADE_TEMPLATE
            self.metric = "llm-explain-with-reference"
        else:
            self.system_prompt = SYSTEM_PROMPT_WITHOUT_KEY_POINT
            self.template = GRADE_TEMPLATE_WITHOUT_KEY_POINT
            self.metric = "llm-explain-without-reference"
    
    def evaluate(self, question, prediction, answer):
        if not prediction:
            return []
        rationale_scores = []
        details = []
        correct_rationale = []
        if self.reference:
            for scoring_point in answer:
                contains_key_point, detail = self.evaluate_single_key_point(question, prediction, scoring_point)
                if contains_key_point is None:
                    return None
                rationale_scores.append(contains_key_point)
                if contains_key_point:
                    correct_rationale.append(scoring_point)
                details.append(detail)
        else:
            contains_key_point, detail = self.evaluate_single_key_point(question, prediction)
            if contains_key_point is None:
                return None
            rationale_scores.append(contains_key_point)
            details.append(detail)

        rationale_score = sum(rationale_scores) / len(rationale_scores) if len(rationale_scores) > 0 else 0
        metric_infos = [{
            "metric_name": self.metric,
            "score": rationale_score,
            "details": details,
            "correct_rationale": correct_rationale if self.reference else None,
            "correct_rationale_num": len(correct_rationale) if self.reference else None
        }]
        logging.debug(f"metric_infos: {metric_infos}")
        return metric_infos

    def evaluate_single_key_point(self, question, prediction, scoring_point=None):
        prompt = self.template.format(question=question,
                                       prediction=prediction,
                                       scoring_point=scoring_point)
        ## reserve for record details
        messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": prompt}
            ]
        # if self.model_name in self.args.commercial_llms:
        #     prompt = messages
        # else:
        #     prompt = self.client.tokenizer.apply_chat_template(messages, tokenize=False)
        response_text = self.client.generate(prompt, system_prompt=self.system_prompt)
        # response_text = resp.choices[0].message.content
        # response_text = resp
        details = {
            "messages": messages,
            "response_text": response_text,
        }
        try:
            extracted = extract_json_from_response(response_text)
            parsed = safe_json_loads(extracted)
            if self.reference:
                details['judge_reason'] = parsed.get("judge_reason", "")
                contains_key_point = parsed.get("contains_key_point", False)
                if isinstance(contains_key_point, str):
                    contains_key_point = contains_key_point.lower() in {"true", "yes", "1"}
                elif isinstance(contains_key_point, (int, float)):
                    contains_key_point = bool(contains_key_point)
                return bool(contains_key_point), details
                # return parsed["contains_key_point"], details
            else:
                details['gold_steps'] = parsed.get("gold_steps", [])
                details['recalled_steps'] = parsed.get("recalled_steps", [])
                print(f"-"*80)
                print(f"parsed: {parsed}")
                print(f"-"*80)
                details['number_of_recalled_steps'] = int(parsed.get("number_of_recalled_steps", 0))
                details['total_number_of_required_steps'] = int(parsed.get("total_number_of_required_steps", 0))
                print(f"-"*80)
                print(f"details['total_number_of_required_steps']: {details['total_number_of_required_steps']}")
                print(f"details['number_of_recalled_steps']: {details['number_of_recalled_steps']}")
                print(f"-"*80)
                if details['total_number_of_required_steps'] > 0:
                    details['final_score'] = details['number_of_recalled_steps'] / details['total_number_of_required_steps']
                else:
                    details['final_score'] = 0
                return details['final_score'], details
        except Exception as e:
            logging.warning(f"[Parse-fail] {e}\n--- Raw ---\n{response_text}\n")
            details["judge_reason"] = f"[Parsing error] {e}"
            # failure fallback
            # return (False, details) if self.reference else (0.0, details)
            return None, details
            

_REGISTRY = {
    "text-metric": TextMetricEvaluator,
    "llm-explain-with-reference":  lambda **kw: LLMJudge(judge_method="llm-explain-with-reference", **kw),
    "llm-explain-without-reference": lambda **kw: LLMJudge(judge_method="llm-explain-without-reference", **kw),
}

def build_evaluator(judge_method, text_metrics=None, **kwargs):
    logging.info(f"Building evaluator: {judge_method}, {text_metrics}, {kwargs}")
    if judge_method == "text-metric":
        return TextMetricEvaluator(metrics=text_metrics)
    else:
        return _REGISTRY[judge_method](**kwargs)
