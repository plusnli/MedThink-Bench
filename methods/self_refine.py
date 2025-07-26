from utils import parse_output

# Initial answer generation prompt
INITIAL_ANSWER_PROMPT = """
Your task is to answer the medical question below.

Question:
{question}

Let's think about this question step by step:
[Please write your reasoning process here]

Based on the above analysis, my final answer is:
[Please provide your answer in the format of "A. option1", "B. option2", etc.]
"""

# Evaluation prompt to critique the current answer's reasoning
EVALUATION_PROMPT = """
You are a critical medical expert. Evaluate the reasoning in the following answer to a medical question.
Focus specifically on the quality of the reasoning that leads to the conclusion.

Question:
{question}

Answer to evaluate:
{answer}

Reasoning from the answer:
{reasoning}

Please provide a detailed evaluation focusing on:
1. Medical accuracy of the reasoning
2. Completeness of the reasoning process
3. Logical structure and flow of reasoning
4. Missing important reasoning steps or considerations
5. Any factual errors in the reasoning

Your evaluation:
"""

# Refinement prompt to improve the answer based on evaluation
REFINEMENT_PROMPT = """
Your task is to improve the reasoning in the previous answer based on the evaluation provided.

Question:
{question}

Previous answer:
{answer}

Previous reasoning:
{reasoning}

Evaluation of previous reasoning:
{evaluation}

Please provide an improved answer that addresses all the reasoning issues mentioned in the evaluation.
Focus on enhancing the step-by-step reasoning process to arrive at the correct conclusion.
Make sure your reasoning is medically accurate, complete, and follows a logical structure.

Let's think about this question step by step:
[Provide your improved reasoning here]

Based on the above analysis, my final answer is:
[Please provide your answer in the format of "A. option1", "B. option2", etc.]
"""


class Self_Refine:
    def __init__(self, args, model, sampling_params):
        self.args = args
        self.model = model
        self.sampling_params = sampling_params
        self.max_iterations = getattr(args, "max_iterations_self_refine", 2)  # Default to 2 iterations if not specified
    
    def _generate_text(self, prompt):
        """Generate text using the configured model."""
        if self.args.model in self.args.commercial_llms:
            response = self.model.responses.create(
                model=self.args.model_id,
                input=prompt,
                max_output_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature
            )
            resp = response.output_text
        else:
            resp = self.model.generate(prompt, self.sampling_params, use_tqdm=False)
            resp = resp[0].outputs[0].text.strip()
        
        return resp

    def _extract_reasoning(self, answer):
        """Extract the reasoning part from an answer."""
        if "Let's think about this question step by step:" in answer:
            parts = answer.split("Let's think about this question step by step:")
            if len(parts) > 1:
                reasoning_part = parts[1]
                if "Based on the above analysis" in reasoning_part:
                    reasoning = reasoning_part.split("Based on the above analysis")[0].strip()
                else:
                    reasoning = reasoning_part.strip()
                return reasoning
        
        # If the standard format isn't found, look for the reasoning portion
        if "Based on the above analysis" in answer:
            reasoning = answer.split("Based on the above analysis")[0].strip()
            return reasoning
        
        # If we can't identify the reasoning clearly, use the whole answer
        return answer

    def _generate_initial_answer(self, question):
        """Generate initial answer to the question."""
        prompt = INITIAL_ANSWER_PROMPT.format(question=question)
        return self._generate_text(prompt)
    
    def _evaluate_answer(self, question, answer, reasoning):
        """Evaluate the current answer with explicit focus on reasoning."""
        prompt = EVALUATION_PROMPT.format(
            question=question,
            answer=answer,
            reasoning=reasoning
        )
        return self._generate_text(prompt)
    
    def _refine_answer(self, question, answer, reasoning, evaluation):
        """Refine answer based on evaluation, focusing on improving reasoning."""
        prompt = REFINEMENT_PROMPT.format(
            question=question,
            answer=answer,
            reasoning=reasoning,
            evaluation=evaluation
        )
        return self._generate_text(prompt)
    
    def _should_stop_refining(self, evaluation):
        """Determine if we should stop refining based on evaluation."""
        positive_indicators = [
            "no issues", "excellent", "accurate", "complete", 
            "well-reasoned", "thorough", "perfect", "optimal",
            "reasoning is sound", "reasoning is clear", "logical flow is appropriate"
        ]
        
        # Check if evaluation contains positive indicators
        for indicator in positive_indicators:
            if indicator.lower() in evaluation.lower():
                return True
        
        return False
    
    def generate(self, question) -> tuple[str, str]:
        """
        Generate answer with self-refinement focused on reasoning.
        
        Returns:
            Tuple of (full response with refinement history, final prediction)
        """
        # Generate initial answer
        current_answer = self._generate_initial_answer(question)
        current_reasoning = self._extract_reasoning(current_answer)
        current_pred = parse_output(current_answer)
        if current_pred is not None and current_pred != "":
            current_pred = current_pred[0]
        
        history = [{
            "iteration": 0,
            "answer": current_answer,
            "reasoning": current_reasoning,
            "prediction": current_pred,
            "evaluation": None
        }]
        
        # Perform refinement iterations
        for i in range(self.max_iterations):
            # Evaluate current answer with focus on reasoning
            evaluation = self._evaluate_answer(question, current_answer, current_reasoning)
            history[-1]["evaluation"] = evaluation
            
            # Check if we should stop refining
            if self._should_stop_refining(evaluation):
                break
                
            # Refine the answer with focus on improving reasoning
            refined_answer = self._refine_answer(question, current_answer, current_reasoning, evaluation)
            refined_reasoning = self._extract_reasoning(refined_answer)
            refined_pred = parse_output(refined_answer)
            if refined_pred is not None and refined_pred != "":
                refined_pred = refined_pred[0]
            
            # Update history
            history.append({
                "iteration": i + 1,
                "answer": refined_answer,
                "reasoning": refined_reasoning,
                "prediction": refined_pred,
                "evaluation": None
            })
            
            # Update current answer and reasoning
            current_answer = refined_answer
            current_reasoning = refined_reasoning
            current_pred = refined_pred
        
        # Compile full response for reference
#         full_resp = f"""
# Question:
# {question}

# Refinement History:
# """
#         for entry in history:
#             full_resp += f"\n--- Iteration {entry['iteration']} ---\n"
#             full_resp += f"Answer:\n{entry['answer']}\n"
#             full_resp += f"Reasoning:\n{entry['reasoning']}\n"
#             full_resp += f"Prediction: {entry['prediction']}\n"
#             if entry["evaluation"]:
#                 full_resp += f"\nEvaluation of Reasoning:\n{entry['evaluation']}\n"
        
#         return full_resp, current_pred
        
        return current_reasoning, current_pred