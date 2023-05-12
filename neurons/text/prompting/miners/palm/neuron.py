# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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

import google.generativeai as genai
import argparse
import bittensor
from typing import List, Dict


class IllegalArgumentError(Exception):
    pass

class PalmMiner( bittensor.BasePromptingMiner ):

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        assert config.palm.api_key != None, 'the miner requires passing --palm.api_key as an argument of the config.'

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument('--palm.api_key', type=str, help='palm api key')
        parser.add_argument('--palm.temperature', type=float, default=0.7, help="Sampling temperature to use, between 0 and 2.")
        parser.add_argument('--palm.context', type=str, default=None, help="...")
        parser.add_argument('--palm.examples', type=..., default=..., help="...")
        parser.add_argument('--palm.candidate_count', type=int, default=1, help="...")
        parser.add_argument('--palm.client', type=..., default=1, help="...")
        parser.add_argument('--palm.top_k', type=float, default=1, help="...")
        parser.add_argument('--palm.top_p', type=float, default=1, help="Nucleus sampling parameter, top_p probability mass.")
        parser.add_argument('--palm.model_name', type=str, default='gpt-3.5-turbo', help="OpenAI model to use for completion.")

    def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ) -> str: pass

    def __init__( self ):
        super( PalmMiner, self ).__init__()
        print ( self.config )
        genai.configure(api_key=self.config.palm.api_key)

        for model in genai.list_models():
            if model.name == self.config.palm.model_name:
                self.model = model
                break
        
        if self.model == None:
            raise IllegalArgumentError(f'Model "{model.self.config.palm.model_name}" not found')


    
    def forward( self, messages: List[Dict[str, str]]  ) -> str:

        resp = genai.chat(
            model = self.model, # google.generativeai.types.Model
            context= self.config.palm.context,
            examples = self.config.palm.examples,
            messages = messages,
            temperature = self.config.palm.temperature,
            candidate_count = self.config.palm.candidate_count,
            top_p = self.config.palm.top_p,
            top_k = self.config.palm.top_k,
            prompt = None,
            client= None
        ).last

        return resp

if __name__ == "__main__":
    bittensor.utils.version_checking()
    PalmMiner().run()
