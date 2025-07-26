from collections import Counter
from utils import parse_output
from .base import BaseMethod
from .cot import ZERO_SHOT_COT


ZERO_SHOT_COT_SC = """
Given the question and the prediction, please think step by step and write a reasoning process for the prediction.

Question:
{question}

Prediction:
{prediction}

Reasoning process:
(Please write your reasoning process here)
"""


class CoT_SC(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        assert self.args.cnt is not None

    def run(self, question) -> tuple[list[str], str]:
        preds = []
        for _ in range(self.args.cnt):
            cot_prompt = ZERO_SHOT_COT.format(question=question)
            resp = self.manager.generate(cot_prompt)
            if not resp:
                continue
            pred = parse_output(resp)[0]
            preds.append(pred)
        counter = Counter(preds)
        final_pred = counter.most_common(1)[0][0] if preds else None
        
        # generate final rationale based on final prediction
        final_rationale_prompt = ZERO_SHOT_COT_SC.format(question=question, prediction=final_pred)
        final_rationale = self.manager.generate(final_rationale_prompt)
        if not final_rationale:
            return None, final_pred
        return final_rationale, final_pred
