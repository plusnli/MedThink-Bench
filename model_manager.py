import os
import logging

import torch
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelManager:
    def __init__(self, args, model_name=None, model_id=None):
        self.args = args
        self.tokenizer = None
        self.model_name = model_name
        self.model_id = model_id
        self.model = self._init_model()
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            seed=args.seed,
            max_tokens=args.max_new_tokens
        )
        
    def _init_model(self):
        """init model"""
        if self.model_name in self.args.commercial_llms:
            if self.model_name in ("gpt-4o", "gpt-4o-mini", "o1", "o3"):
                return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            elif 'claude' in self.model_name:
                return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            elif 'gemini' in self.model_name:
                return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            elif 'DeepSeek' in self.model_name:
                return OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
            else:
                raise ValueError(f"Model {self.model_name} not supported yet.")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            # For Baichuan-M1-14B: use vanilla transformers
            if "Baichuan-M1-14B" in self.model_name or "Baichuan-M1-14B" in self.model_id:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype = torch.bfloat16).cuda()
                return self.model

            # For other open-source models: use vLLM
            return LLM(
                model=self.model_id,
                trust_remote_code=True,
                gpu_memory_utilization=self.args.gpu_utilization,
                tensor_parallel_size=self.args.tensor_parallel,
                max_model_len=self.args.max_new_tokens,
                quantization=self.args.quantization,
                seed=self.args.seed
            )

    def generate(self, prompt, system_prompt=None) -> str:
        """unified generation interface"""
        try:
            if self.model_name in self.args.commercial_llms:
                if self.model_name in ("gpt-4o", "gpt-4o-mini", "o1"):
                    if self.args.gen:
                        response = self.model.responses.create(
                            model=self.model_id,
                            input=prompt,
                            max_output_tokens=self.args.max_new_tokens,
                            temperature=self.args.temperature
                        )
                        resp = response.output_text
                    elif self.args.eval:
                        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
                        response = self.model.chat.completions.create(
                            model=self.model_id,
                            messages=messages,
                            **vars(self.sampling_params)
                        )
                        resp = response.choices[0].message.content
                    else:
                        raise ValueError(f"Unknown run mode")
                elif self.model_name in ("o3"):  # o3 does not support temperature
                    if self.args.eval:
                        sampling_kwargs = {k: v for k, v in vars(self.sampling_params).items() if k != "temperature"}
                        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
                        response = self.model.chat.completions.create(
                            model=self.model_id,
                            messages=messages,
                            **sampling_kwargs
                        )
                    response = self.model.responses.create(
                        model=self.model_id,
                        input=prompt,
                        max_output_tokens=self.args.max_new_tokens
                    )
                    resp = response.output_text
                elif 'claude' in self.model_name:
                    messages = [{"role": "user", "content": prompt}]
                    message = self.model.messages.create(
                        model=self.model_id,
                        messages=messages,
                        system=system_prompt,
                        max_tokens=self.args.max_new_tokens,
                        temperature=self.args.temperature
                    )
                    resp = message.content[0].text
                elif 'gemini' in self.model_name:
                    config_kwargs = {
                        "max_output_tokens": self.args.max_new_tokens,
                        "temperature": self.args.temperature
                    }

                    if self.args.eval:  # eval mode needs system prompt
                        config_kwargs["system_instruction"] = system_prompt

                    response = self.model.models.generate_content(
                        model=self.model_id,
                        contents=prompt,
                        config=types.GenerateContentConfig(**config_kwargs)
                    )
                    resp = response.text
                elif 'DeepSeek' in self.model_name:
                    if self.args.eval:
                        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
                    else: # gen 
                        messages = [{"role": "user", "content": prompt}]
                    response = self.model.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        stream=False,
                        max_tokens=self.args.max_new_tokens,
                        temperature=self.args.temperature
                    )
                    resp = response.choices[0].message.content
            else:
                if "Baichuan-M1-14B" in self.model_name or "Baichuan-M1-14B" in self.model_id:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt = True)
                    inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                    generated_ids = self.model.generate(**inputs, max_new_tokens=self.args.max_new_tokens)
                    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
                    resp = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    # resp = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                else:
                    if self.args.eval:
                        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
                        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
                    resp = self.model.generate(prompt, self.sampling_params, use_tqdm=False)
                    resp = resp[0].outputs[0].text.strip()
            return resp
        except Exception as e:
            logging.error(f"Error in model generation: {str(e)}")
            return str(e)
