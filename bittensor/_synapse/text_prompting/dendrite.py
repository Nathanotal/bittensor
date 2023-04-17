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
import grpc
import json
import torch
import asyncio
import bittensor
from typing import Callable, List, Dict, Union

class TextPromptingForwardCall( bittensor.DendriteCall ):

    name: str = "text_prompting_forward"
    is_forward: bool = True
    completion: str = "" # To be filled.

    def __init__(
        self,
        dendrite: bittensor.TextPromptingDendrite,
        messages: List[str],
        roles: List[str],
        timeout: float = bittensor.__blocktime__,
    ):
        super().__init__( dendrite = dendrite, timeout = timeout )
        self.messages = messages
        self.roles = roles
        self.packed_messages = [json.dumps({"role": role, "content": message}) for role, message in zip(self.roles, self.messages)]

    def callable( self ) -> Callable:
        bittensor.grpc.TextPromptingStub( self.dendrite.receptor.channel ).Forward

    def get_request_proto( self ) -> bittensor.proto.ForwardTextPromptingRequest:
        return bittensor.ForwardTextPromptingRequest( timeout = self.timeout, messages = self.packed_messages )
    
    def apply_response_proto( self, response_proto: bittensor.ForwardTextPromptingResponse ):
        self.completion = response_proto.response
        
    def get_inputs_shape(self) -> Union[torch.Size, None]: 
        return torch.Size( [len(message) for message in self.packed_messages] )

    def get_outputs_shape(self) -> Union[torch.Size, None]:
        return torch.Size([ len(self.completion) ] )
    
class TextPromptingBackwardCall( bittensor.DendriteCall ):

    name: str = "text_prompting_backward"
    is_forward: bool = False
    
    def __init__(
        self,
        dendrite: bittensor.TextPromptingDendrite,
        completion: str,
        messages: List[str],
        roles: List[str],
        rewards: Union[ List[float], torch.FloatTensor ],
        timeout: float = bittensor.__blocktime__,
    ):
        super().__init__( dendrite = dendrite, timeout = timeout )
        self.messages = messages
        self.roles = roles
        self.completion = completion
        self.rewards = rewards if not isinstance( rewards, torch.FloatTensor ) else rewards.tolist()
        self.packed_messages = [ json.dumps({"role": role, "content": message}) for role, message in zip(self.roles, self.messages)]

    def callable( self ) -> Callable:
        bittensor.grpc.TextPromptingStub( self.dendrite.receptor.channel ).Backward

    def get_request_proto( self ) -> bittensor.proto.ForwardTextPromptingRequest:
        return bittensor.BackwardTextPromptingRequest( messages = self.packed_messages, response = self.completion, rewards = self.rewards )
    
    def apply_response_proto( self, response_proto: bittensor.ForwardTextPromptingResponse ):
        pass
        
    def get_inputs_shape(self) -> Union[torch.Size, None]:
        return torch.Size( [len(message) for message in self.packed_messages] )

    def get_outputs_shape(self) -> Union[torch.Size, None]:
        return torch.Size( [0] )


class TextPromptingDendrite( bittensor.Dendrite ):

    def get_stub(self, channel) -> Callable:
        return bittensor.grpc.TextPromptingStub(channel)

    def forward(
            self,
            roles: List[ str ] ,
            messages: List[ str ],
            timeout: float = bittensor.__blocktime__,
            return_call:bool = True,
        ) -> Union[ str, "bittensor.TextPromptingForwardCall" ]:
        forward_call = bittensor.TextPromptingForwardCall(
            dendrite = self, 
            messages = messages,
            roles = roles,
            timeout = timeout,
        )
        loop = asyncio.get_event_loop()
        response_call = loop.run_until_complete( self.apply( dendrite_call = forward_call ) )
        if return_call: return response_call
        else: return response_call.completion
    
    async def async_forward(
        self,
        roles: List[ str ],
        messages: List[ str ],
        timeout: float = bittensor.__blocktime__,
        return_call: bool = True,
    ) -> Union[ str, "bittensor.TextPromptingForwardCall" ]:
        forward_call = bittensor.TextPromptingForwardCall(
            dendrite = self, 
            messages = messages,
            roles = roles,
            timeout = timeout,
        )
        forward_call = await self.apply( dendrite_call = forward_call )
        if return_call: return forward_call
        else: return forward_call.completion

    def backward(
            self,
            roles: List[ str ],
            messages: List[ str ],
            completion: str,
            rewards: Union[ List[ float], torch.FloatTensor ],
            timeout: float = bittensor.__blocktime__,
        ) -> "bittensor.TextPromptingBackwardCall":
        backward_call = bittensor.TextPromptingBackwardCall(
            completion = completion,
            messages = messages,
            roles = roles,
            rewards = rewards
            timeout = timeout,
        )
        loop = asyncio.get_event_loop()
        loop.run_until_complete( self.apply( dendrite_call = backward_call ) )

    async def async_backward(
        self,
        roles: List[ str ],
        messages: List[ str ],
        completion: str,        
        rewards: Union[ List[ float], torch.FloatTensor ],
        timeout: float = bittensor.__blocktime__,
    ) -> "bittensor.TextPromptingBackwardCall":
        backward_call = bittensor.TextPromptingBackwardCall(
            completion = completion,
            messages = messages,
            roles = roles,
            rewards = rewards,
            timeout = timeout,
        )
        await self.apply( dendrite_call = backward_call ) 




