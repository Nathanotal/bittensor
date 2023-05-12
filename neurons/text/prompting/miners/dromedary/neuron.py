# TODO: LICENSE

import bittensor
import argparse
from typing import List, Dict
from transformers import LlamaForCausalLM, LlamaTokenizer # Maybe not
# import torch


class DromedaryMiner(bittensor.BasePromptingMiner):

    @classmethod
    def check_config(cls, config: 'bittensor.Config'):
        pass

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        # TODO: implement
        ...
        parser.add_argument('--dromedary.model_name', type=str,
                            required=True, help='Name/path of model to load')
        parser.add_argument( '--dromedary.device', type=str, help='Device to load model', default="cuda" )
        
        parser.add_argument( '--dromedary.nproc_per_node', type=..., help='...', default="..." )
        parser.add_argument( '--dromedary.nnodes', type=..., help='...', default="..." )
        parser.add_argument( '--dromedary.node_rank', type=..., help='...', default="..." )
        parser.add_argument( '--dromedary.master_addr', type=..., help='...', default="..." )
        parser.add_argument( '--dromedary.master_port', type=..., help='...', default="..." )
        parser.add_argument( '--dromedary.ckpt_dir', type=..., help='...', default="..." )
        parser.add_argument( '--dromedary.tokenizer_path', type=..., help='...', default="..." )
        parser.add_argument( '--dromedary.max_seq_len', type=..., help='...', default="..." )
        parser.add_argument( '--dromedary.max_batch_size', type=..., help='...', default="..." )
        parser.add_argument( '--dromedary.meta_prompt_file', type=..., help='...', default="..." )
        # parser.add_argument( '--dromedary.max_new_tokens', type=int, help='Max tokens for model output.', default=256 )
        # parser.add_argument( '--dromedary.temperature', type=float, help='Sampling temperature of model', default=0.5 )
        # parser.add_argument( '--dromedary.do_sample', action='store_true', default=False, help='Whether to use sampling or not (if not, uses greedy decoding).' )
        # parser.add_argument( '--dromedary.do_prompt_injection', action='store_true', default=False, help='Whether to use a custom "system" prompt instead of the one sent by bittensor.' )
        # parser.add_argument( '--dromedary.system_prompt', type=str, help='What prompt to replace the system prompt with', default= "BEGINNING OF CONVERSATION: " )

    def __init__(self):
        super(DromedaryMiner, self).__init__()
        print(self.config)  # TODO: remove?

        # TODO: implement

        bittensor.logging.info(
            'Loading ' + str(self.config.dromedary.model_name))

        # use_fast=False
        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.config.dromedary.model_name)
        # torch_dtype=torch.float16, low_cpu_mem_usage=True
        self.model = LlamaForCausalLM.from_pretrained(
            self.config.dromedary.model_name)

        bittensor.logging.info('Model loaded!')

        if self.config.dromedary.device != "cpu":
            self.model = self.model.to(self.config.dromedary.device)

        # ------------------
        # set -e
        # set -x

        # export PYTHONPATH="$PWD:$PYTHONPATH"
        # export CUDA_VISIBLE_DEVICES=0,1
        # export MODEL_DIR="/path/to/your/model/dir"
        # export OMP_NUM_THREADS=2
        # export GPUS_PER_NODE=2
        # export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
        # export MASTER_PORT=9901
        # export TOTAL_NUM_GPUS=$(( $SLURM_NNODES * $GPUS_PER_NODE ))

        # N_SHARDS=2
        # CKPT_NAME="dromedary-65b-lora-final"

        # torchrun --nproc_per_node $GPUS_PER_NODE \
        # --nnodes $SLURM_NNODES \
        # --node_rank $SLURM_PROCID \
        # --master_addr $MASTER_ADDR \
        # --master_port $MASTER_PORT \
        # run_chatbot_demo.py \
        # --ckpt_dir $MODEL_DIR/$CKPT_NAME-${N_SHARDS}shards \
        # --tokenizer_path $MODEL_DIR/tokenizer.model \
        # --max_seq_len 2048 \
        # --max_batch_size 1 \
        # --meta_prompt_file "../prompts/inference_prompts/dromedary_verbose_prompt.txt"

    def _process_history(self, history: List[str]) -> str:
        # TODO: implement

        processed_history = ''

        # if self.config.dromedary.do_prompt_injection:
        #     processed_history += self.config.dromedary.system_prompt

        # for message in history:
        #     if message['role'] == 'system':
        #         if not self.config.dromedary.do_prompt_injection or message != history[0]:
        #             processed_history += '' + message['content'].strip() + ' '

        #     if message['role'] == 'Assistant':
        #         # No blankspace after GPT: since that is where generation starts.
        #         processed_history += 'GPT:' + \
        #             message['content'].strip() + '</s>'
        #     if message['role'] == 'user':
        #         processed_history += 'USER: ' + \
        #             message['content'].strip() + ' '

        return processed_history

    def forward(self, messages: List[Dict[str, str]]) -> str:
        # TODO: implement

        generation = None

        # history = self._process_history(messages)
        # prompt = history + "GPT:"
        # input_ids = self.tokenizer.encode(
        #     prompt, return_tensors="pt").to(self.config.dromedary.device)
        # output = self.model.generate(
        #     input_ids,
        #     max_length=input_ids.shape[1] + self.config.dromedary.max_new_tokens,
        #     temperature=self.config.dromedary.temperature,
        #     do_sample=self.config.dromedary.do_sample,
        #     pad_token_id=self.tokenizer.eos_token_id,
        # )
        # generation = self.tokenizer.decode(
        #     output[0][input_ids.shape[1]:], skip_special_tokens=True)

        # Logging input and generation if debugging is active

        bittensor.logging.debug("Message: " + str(messages))
        bittensor.logging.debug("Generation: " + str(generation))
        return generation


if __name__ == "__main__":
    bittensor.utils.version_checking()
    DromedaryMiner().run()
