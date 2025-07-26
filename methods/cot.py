import json
from utils import parse_output
from .base import BaseMethod


ZERO_SHOT_COT = """
You are an experienced doctor. Your task is to answer the question below.

Question:
{question}

Let’s think about this question step by step, and you should provide the detailed reasoning process. 
(Please write your reasoning process here.)

Based on the above reasoning analysis, please give the final answer here. 
(Notably, the final answer should be quoted with Square Brackets. For example, ["A. option1"], ["B. option2"], etc.)
"""

FEW_SHOT_COT = """
You are an experienced doctor. Your task is to answer the question below.

{examples_str}
Question:
{question}

Let’s think about this question step by step, and you should provide the detailed reasoning process. 
(Please write your reasoning process here.)

Based on the above reasoning analysis, please give the final answer here. 
(Notably, the final answer should be quoted with Square Brackets. For example, ["A. option1"], ["B. option2"], etc.)
"""

# ZERO_SHOT_COT_RATIONALE = """
# Your task is to answer the question below.

# Question:
# {question}

# Let's think about this question step by step:
# [Please write your reasoning process here]
# """

# ZERO_SHOT_COT_PRED = """
# Your task is to answer the question below.

# Question:
# {question}

# Rationale:
# {rationale}

# Based on the above analysis, my final answer is:
# [Please provide your answer in the format of "A. option1", "B. option2", etc.]
# """


class CoT(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        self.few_shot = args.few_shot
        self.shot_num = args.shot_num
        if args.few_shot:
            self.examples_str = "Here are some examples:\n" if args.shot_num > 1 else ""
            self.examples = []
            with open("data_20_demonstrations_0620.json", "r") as f:
                self.examples = json.load(f)
            for i, example in enumerate(self.examples[:self.shot_num]):
                explanation_steps = example['Scoring_Points_expert']
                explanation_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(explanation_steps)])
                self.examples_str += f"Example {i+1}: {example['question']}\nAnswer: [{example['answer']}]\nExplanation: {explanation_str}\n\n"

    def run(self, question) -> tuple[str, str]:
        if self.few_shot:
            cot_prompt = FEW_SHOT_COT.format(question=question, examples_str=self.examples_str)
        else:
            cot_prompt = ZERO_SHOT_COT.format(question=question)
        cot = self.manager.generate(cot_prompt)
        if not cot:
            return None, None
        pred = parse_output(cot)[0]
        return cot, pred
