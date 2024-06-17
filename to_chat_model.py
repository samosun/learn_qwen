from typing import Tuple, List, Union, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from transformers import logging
from transformers.generation import LogitsProcessor

logger = logging.get_logger(__name__)

# Types.
HistoryType = List[Tuple[str, str]]
TokensType = List[int]
BatchTokensType = List[List[int]]


# step 1 加载模型

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("/media/cfs/gengdongyang/20231207_longterm-fcst/aliendao/Qwen1.5-32B-Chat-GPTQ-Int4")
model =  AutoModelForCausalLM.from_pretrained("/media/cfs/gengdongyang/20231207_longterm-fcst/aliendao/Qwen1.5-32B-Chat-GPTQ-Int4",
                                             device_map='cuda:0').eval()
model = model.eval()

# step 2 为model 增加chat 方法，实现https://huggingface.co/Qwen/Qwen-72B-Chat/blob/main/modeling_qwen.py 中chat方法
_ERROR_BAD_CHAT_FORMAT = """\
We detect you are probably using the pretrained model (rather than chat model) for chatting, since the chat_format in generation_config is not "chatml".
If you are directly using the model downloaded from Huggingface, please make sure you are using our "Qwen/Qwen-7B-Chat" Huggingface model (rather than "Qwen/Qwen-7B") when you call model.chat().
我们检测到您可能在使用预训练模型（而非chat模型）进行多轮chat，因为您当前在generation_config指定的chat_format，并未设置为我们在对话中所支持的"chatml"格式。
如果您在直接使用我们从Huggingface提供的模型，请确保您在调用model.chat()时，使用的是"Qwen/Qwen-7B-Chat"模型（而非"Qwen/Qwen-7B"预训练模型）。
"""

_SENTINEL = object()
_ERROR_STREAM_IN_CHAT = """\
Pass argument `stream` to model.chat() is buggy, deprecated, and marked for removal. Please use model.chat_stream(...) instead of model.chat(..., stream=True).
向model.chat()传入参数stream的用法可能存在Bug，该用法已被废弃，将在未来被移除。请使用model.chat_stream(...)代替model.chat(..., stream=True)。
"""

_ERROR_INPUT_CPU_QUERY_WITH_FLASH_ATTN_ACTIVATED = """\
We detect you have activated flash attention support, but running model computation on CPU. Please make sure that your input data has been placed on GPU. If you actually want to run CPU computation, please following the readme and set device_map="cpu" to disable flash attention when loading the model (calling AutoModelForCausalLM.from_pretrained).
检测到您的模型已激活了flash attention支持，但正在执行CPU运算任务。如使用flash attention，请您确认模型输入已经传到GPU上。如果您确认要执行CPU运算，请您在载入模型（调用AutoModelForCausalLM.from_pretrained）时，按照readme说法，指定device_map="cpu"以禁用flash attention。
"""


def get_stop_words_ids(chat_format, tokenizer):
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids


def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(role) + nl_tokens + tokenizer.encode(content)

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens


def _decode_chatml(
    tokens: List[int],
    *,
    stop_words: List[str],
    eod_token_ids: List[int],
    tokenizer: PreTrainedTokenizer,
    raw_text_len: int,
    context_length: int,
    verbose: bool = False,
    return_end_reason: bool = False,
    errors: str='replace'
):
    end_reason = f"Gen length {len(tokens)}"
    eod_token_idx = context_length
    for eod_token_idx in range(context_length, len(tokens)):
        if tokens[eod_token_idx] in eod_token_ids:
            end_reason = f"Gen {tokenizer.decode([tokens[eod_token_idx]])!r}"
            break

    trim_decode_tokens = tokenizer.decode(tokens[:eod_token_idx], errors=errors)[raw_text_len:]
    if verbose:
        print("\nRaw Generate w/o EOD:", tokenizer.decode(tokens, errors=errors)[raw_text_len:])
        print("\nRaw Generate:", trim_decode_tokens)
        print("\nEnd Reason:", end_reason)
    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, "").strip()
    trim_decode_tokens = trim_decode_tokens.strip()
    if verbose:
        print("\nGenerate:", trim_decode_tokens)

    if return_end_reason:
        return trim_decode_tokens, end_reason
    else:
        return trim_decode_tokens


def decode_tokens(
    tokens: Union[torch.LongTensor, TokensType],
    tokenizer: PreTrainedTokenizer,
    raw_text_len: int,
    context_length: int,
    chat_format: str,
    verbose: bool = False,
    return_end_reason: bool = False,
    errors: str="replace",
) -> str:
    if torch.is_tensor(tokens):
        tokens = tokens.cpu().numpy().tolist()

    if chat_format == "chatml":
        return _decode_chatml(
            tokens,
            stop_words=[],
            eod_token_ids=[tokenizer.im_start_id, tokenizer.im_end_id],
            tokenizer=tokenizer,
            raw_text_len=raw_text_len,
            context_length=context_length,
            verbose=verbose,
            return_end_reason=return_end_reason,
            errors=errors,
        )
    elif chat_format == "raw":
        return _decode_default(
            tokens,
            stop_words=["<|endoftext|>"],
            eod_words=["<|endoftext|>"],
            tokenizer=tokenizer,
            raw_text_len=raw_text_len,
            verbose=verbose,
            return_end_reason=return_end_reason,
            errors=errors,
        )
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

class ToChatModel:
    def __init__(self, model):
        self.model = model
    def chat(
        self,
        tokenizer,
        query: str,
        history=None,
        system: str = "You are a helpful assistant.",
        stop_words_ids = None,
        generation_config = None,
        **kwargs,
    ) :
        generation_config = generation_config if generation_config is not None else self.model.generation_config
        generation_config.chat_format = 'chatml'
        generation_config.max_new_tokens = 2048
        generation_config.max_window_size = 30720
        assert generation_config.chat_format == 'chatml', _ERROR_BAD_CHAT_FORMAT
        if history is None:
            history = []
        else:
            # make a copy of the user's input such that is is left untouched
            history = copy.deepcopy(history)

        if stop_words_ids is None:
            stop_words_ids = []

        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = generation_config.max_window_size
        raw_text, context_tokens = make_context(
            tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=generation_config.chat_format,
        )

        stop_words_ids.extend(get_stop_words_ids(
            generation_config.chat_format, tokenizer
        ))
        input_ids = torch.tensor([context_tokens]).to(self.model.device)
        outputs = self.model.generate(
                    input_ids,
                    # stop_words_ids=stop_words_ids,
                    return_dict_in_generate=False,
                    generation_config=generation_config,
                    **kwargs,
                )

        response = decode_tokens(
            outputs[0],
            tokenizer,
            raw_text_len=len(raw_text),
            context_length=len(context_tokens),
            chat_format=generation_config.chat_format,
            verbose=False,
            errors='replace'
        )

        # as history is a copy of the user inputs,
        # we can always return the new turn to the user.
        # separating input history and output history also enables the user
        # to implement more complex history management
        history.append((query, response))

        return response, history

if __name__ == '__main__':
  tokenizer.im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
  tokenizer.im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
  
  ToChatModel(model).chat(tokenizer, query='你好')
  #('你好！很高兴为你提供帮助。', [('你好', '你好！很高兴为你提供帮助。')]), 可以用于实现openai_api.py中chat方法
