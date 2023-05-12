# The MIT License (MIT)
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import argparse
import bittensor
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM


class DromedaryMiner(bittensor.BasePromptingMiner):

    @classmethod
    def check_config(cls, config: 'bittensor.Config'):
        pass

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--dromedary.model_name', type=str,
                            required=True, help='Name/path of model to load')
        parser.add_argument('--dromedary.gptq_path', type=str,
                            required=True, help='Path to quantized weights, requires safetensor to be set')
        parser.add_argument('--dromedary.device', type=str,
                            help='Device to load model', default="cuda")
        parser.add_argument('--dromedary.max_new_tokens', type=int,
                            help='Max tokens for model output.', default=256)
        parser.add_argument('--dromedary.temperature', type=float,
                            help='Sampling temperature of model', default=0.5)
        parser.add_argument('--dromedary.do_sample', action='store_true', default=False,
                            help='Whether to use sampling or not (if not, uses greedy decoding).')
        parser.add_argument('--dromedary.do_prompt_injection', action='store_true', default=False,
                            help='Whether to use a custom "system" prompt instead of the one sent by bittensor.')
        parser.add_argument('--dromedary.system_prompt', type=str,
                            help='What prompt to replace the system prompt with', default="BEGINNING OF CONVERSATION: ")
        parser.add_argument('--dromedary.internal_thought', required=False,
                            help='Whether to use internal thought or not', action='store_true', default=False)
        parser.add_argument('--dromedary.load_using_safetensors', required=False,
                            help='Whether to load the model using safetensors or not', action='store_true', default=False)

    def __init__(self):
        super(DromedaryMiner, self).__init__()
        print(self.config)

        bittensor.logging.info(
            'Loading ' + str(self.config.dromedary.model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.dromedary.model_name, use_fast=False)
        self.model = self.get_model()
        
        bittensor.logging.info('Model loaded!')

        if self.config.dromedary.device != "cpu":
            self.model = self.model.to(self.config.dromedary.device)
            
    def get_model(self):
        if self.config.dromedary.load_using_safetensors:
            from auto_gptq import AutoGPTQForCausalLM
            return AutoGPTQForCausalLM.from_quantized( self.config.dromedary.gptq_path, model_basename = self.config.dromedary.model_name, device="cuda:0", use_triton=False, use_safetensors=True)

        else:
            return AutoModelForCausalLM.from_pretrained(
            self.config.dromedary.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    def _process_history(self, history: List[str]) -> str:
        processed_history = ''

        if self.config.dromedary.do_prompt_injection:
            processed_history += self.config.dromedary.system_prompt

        for message in history:
            if message['role'] == 'system':
                if not self.config.dromedary.do_prompt_injection or message != history[0]:
                    processed_history += '' + message['content'].strip() + ' '

            if message['role'] == 'assistant':
                processed_history += '\n\n### Dromedary: ' + \
                    message['content'].strip() + ' '
            if message['role'] == 'user':
                processed_history += '\n\n### User: ' + \
                    message['content'].strip() + ' '

        return processed_history

    def forward(self, messages: List[Dict[str, str]]) -> str:
        history = self._process_history(messages)
        if self.config.dromedary.internal_thought:
            prompt = history + '\n\n### Dromedary (internal thought):'
        else:
            prompt = history + "\n\n### Dromedary:"
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt").to(self.config.dromedary.device)
        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] +
            self.config.dromedary.max_new_tokens,
            temperature=self.config.dromedary.temperature,
            do_sample=self.config.dromedary.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generation = self.tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True)
        if self.config.dromedary.internal_thought:
            bittensor.logging.debug(
                "Generation with internal thought: " + str(generation))
            generation = generation.split("### Dromedary:")[-1]

        # Logging input and generation if debugging is active
        bittensor.logging.debug("Message: " + str(messages))
        bittensor.logging.debug("Generation: " + str(generation))
        return generation


if __name__ == "__main__":
    bittensor.utils.version_checking()
    DromedaryMiner().run()
