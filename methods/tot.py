from collections import Counter
from utils import parse_output
from .base import BaseMethod

# Depth is 2 now. The prompts should be changed accordingly if you want to change the depth.
# Width is 1 now.


# Prompt to generate initial thoughts/plans
THOUGHT_GENERATION_PROMPT = """
Your task is to generate plans for solving the medical question below.
Generate a detailed plan for how you would approach this question.

Question:
{question}

Plan:
"""

# Prompt to generate analyses based on the selected plan
ANALYSIS_GENERATION_PROMPT = """
Your task is to analyze the medical question below using the given plan.
Generate a detailed analysis that follows this plan to solve the question.

Question:
{question}

Plan:
{selected_plan}

Analysis:
"""

# Prompt to vote for the best option
VOTE_PROMPT = """
Analyze the choices below, then conclude which is most promising for answering the medical question.

Question:
{question}

Options:
{options}

Based on a careful analysis of all options, the most promising one is:
[Write the number of the most promising option (1, 2, 3, 4, 5, etc)]
"""

# Final answer generation prompt
FINAL_ANSWER_PROMPT = """
Based on the question and the analysis below, provide your final answer.

Question:
{question}

Analysis:
{selected_analysis}

My final answer is:
[Please provide your answer in the format of "A. option1", "B. option2", etc.]
"""

class ToT(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        self.k = args.tot_k  # Number of thoughts to generate at each step
        
    def _vote_for_best(self, question, options):
        """Use voting to select the best option."""
        votes = []
        formatted_options = ""
        for i, option in enumerate(options, 1):
            formatted_options += f"Option {i}:\n{option}\n\n"
        
        # Generate 5 votes
        for _ in range(5):
            vote_prompt = VOTE_PROMPT.format(question=question, options=formatted_options)
            vote_response = self.manager.generate(vote_prompt)
            
            # Extract the vote (looking for numbers 1-5)
            for char in vote_response:
                if char in "12345":
                    votes.append(int(char))
                    break
        
        # Count votes and find the most common
        vote_counts = Counter(votes)
        if not vote_counts:
            # If no valid votes, choose randomly from options
            import random
            return random.choice(options)
        
        # Get the most common vote (if tied, take the first one)
        most_common_vote = vote_counts.most_common(1)[0][0]
        # Convert to 0-indexed for list access
        return options[most_common_vote - 1] if 1 <= most_common_vote <= len(options) else options[0]
    
    def run(self, question) -> tuple[str, str]:
        # Step 1: Generate k initial plans
        plans = []
        for _ in range(self.k):
            plan_prompt = THOUGHT_GENERATION_PROMPT.format(question=question)
            plan = self.manager.generate(plan_prompt)
            plans.append(plan)
        
        # Vote for the best plan
        selected_plan = self._vote_for_best(question, plans)
        
        # Step 2: Generate k analyses based on the selected plan
        analyses = []
        for _ in range(self.k):
            analysis_prompt = ANALYSIS_GENERATION_PROMPT.format(
                question=question, 
                selected_plan=selected_plan
            )
            analysis = self.manager.generate(analysis_prompt)
            analyses.append(analysis)
        
        # Vote for the best analysis
        selected_analysis = self._vote_for_best(question, analyses)
        
        # Generate final answer based on the best analysis
        final_prompt = FINAL_ANSWER_PROMPT.format(
            question=question,
            selected_analysis=selected_analysis
        )
        resp = self.manager.generate(final_prompt)
        
        # Extract the final answer
        pred = parse_output(resp)
        if pred is not None and pred != "":
            pred = pred[0]
            
        # Compile full response for reference
#         full_resp = f"""
# Question:
# {question}

# Selected Plan:
# {selected_plan}

# Selected Analysis:
# {selected_analysis}

# Final Answer:
# {resp}
# """
        
        return selected_analysis, pred