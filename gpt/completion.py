import json
import logging
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from typing import Optional

import requests

DEFAULT_USER = "gpt-cli"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

LOGGER = logging.getLogger("gpt.completion")


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def openai_request(data) -> tuple[int, dict]:
    """
    Make a request to the OpenAI API.
    This is equalant to openai.ChatCompletion.create(**data)
    """

    session = requests.Session()
    response = session.request(
        method="post",
        url="https://api.openai.com/v1/chat/completions",
        headers={
            "X-OpenAI-Client-User-Agent": '{"bindings_version": "0.27.0", "httplib": "requests", "lang": "python", "lang_version": "3.10.9", "platform": "Linux-6.1.4-arch1-1-x86_64-with-glibc2.36", "publisher": "openai", "uname": "Linux 6.1.4-arch1-1 #1 SMP PREEMPT_DYNAMIC Sat, 07 Jan 2023 15:10:07 +0000 x86_64 "}',
            "User-Agent": "OpenAI/v1 PythonBindings/0.27.0",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(data, cls=CustomJSONEncoder),
        files=None,
        stream=False,
        timeout=600,
    )
    return response.status_code, response.json()


@dataclass
class ChatMessage:
    """
    A chat message sent to or received from the OpenAI API.
    """

    role: str
    """
    The role of the message. Either "user" or "assistant".
    """

    content: str
    """
    The content of the message.
    """


@dataclass
class ChatCompletionRequest:
    """
    Creates a completion for the chat message
    """

    model: str
    """
    ID of the model to use. Currently, only gpt-3.5-turbo and gpt-3.5-turbo-0301 are supported.
    """

    messages: list[ChatMessage]
    """
    The messages to generate chat completions for, in the chat format.

    https://platform.openai.com/docs/guides/chat/introduction
    """

    temperature: Optional[float] = 1
    """
    What sampling temperature to use, between 0 and 2. Higher values
    like 0.8 will make the output more random, while lower values like
    0.2 will make it more focused and deterministic.

    We generally recommend altering this or top_p but not both.
    """

    top_p: Optional[float] = 1
    """
    An alternative to sampling with temperature, called nucleus sampling,
    where the model considers the results of the tokens with top_p
    probability mass. So 0.1 means only the tokens comprising the top 10%
    probability mass are considered.

    We generally recommend altering this or temperature but not both.
    """

    n: Optional[int] = 1
    """
    How many completions to generate for each prompt.

    Note: Because this parameter generates many completions, it can
    quickly consume your token quota. Use carefully and ensure that you
    have reasonable settings for max_tokens and stop.
    """

    stream: Optional[bool] = False
    """
    Whether to stream back partial progress. If set, tokens will be sent as
    data-only server-sent events as they become available, with the stream
    terminated by a data: [DONE] message.

    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """

    stop: Optional[str] | list[str] | None = None
    """
    Up to 4 sequences where the API will stop generating further tokens. The
    returned text will not contain the stop sequence.
    """

    max_tokens: Optional[int] = None
    """
    The maximum number of tokens to generate in the completion.

    The token count of your prompt plus max_tokens cannot exceed the
    model's context length. Most models have a context length of 2048
    tokens (except for the newest models, which support 4096).

    https://platform.openai.com/tokenizer
    """

    presence_penalty: Optional[float] = 0
    """
    Number between -2.0 and 2.0. Positive values penalize new tokens based on
    whether they appear in the text so far, increasing the model's likelihood
    to talk about new topics.

    https://platform.openai.com/docs/api-reference/parameter-details
    """

    frequency_penalty: Optional[float] = 0
    """
    Number between -2.0 and 2.0. Positive values penalize new tokens based
    on their existing frequency in the text so far, decreasing the model's
    likelihood to repeat the same line verbatim.

    https://platform.openai.com/docs/api-reference/parameter-details
    """

    logit_bias: Optional[dict] = field(default_factory=dict)
    """
    Modify the likelihood of specified tokens appearing in the completion.

    Accepts a json object that maps tokens (specified by their token
    ID in the tokenizer) to an associated bias value from -100 to
    100. Mathematically, the bias is added to the logits generated by
    the model prior to sampling. The exact effect will vary per model,
    but values between -1 and 1 should decrease or increase likelihood
    of selection; values like -100 or 100 should result in a ban or
    exclusive selection of the relevant token.
    """

    user: Optional[str] = DEFAULT_USER
    """
    A unique identifier representing your end-user, which can help OpenAI
    to monitor and detect abuse.

    https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids
    """

    def __str__(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class ChatCompletionResponse:
    """
    Response from the OpenAI API for a chat completion request.
    """

    @dataclass
    class Choices:
        message: ChatMessage
        index: int
        finish_reason: str

        def __post_init__(self):
            if isinstance(self.message, dict):
                self.message = ChatMessage(**self.message)

    @dataclass
    class Usage:
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int

    id: str
    object: str
    created: datetime
    model: str
    choices: list[Choices]
    usage: Usage

    def __post_init__(self):
        if isinstance(self.created, int):
            self.created = datetime.fromtimestamp(self.created)

        if isinstance(self.choices, list):
            self.choices = [
                self.Choices(**choice)
                for choice in self.choices
                if isinstance(choice, dict)
            ]

        if isinstance(self.usage, dict):
            self.usage = self.Usage(**self.usage)

    @property
    def message(self) -> ChatMessage:
        return self.choices[0].message


class ChatCompletionError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message


class ChatCompletion:
    @classmethod
    def create(cls, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Create a chat completion.

        https://platform.openai.com/docs/guides/chat
        """
        LOGGER.debug("Creating chat completion: %s", request)
        status, data = openai_request(request)
        if status == 200:
            return ChatCompletionResponse(**data)
        else:
            LOGGER.error("Error creating chat completion: %s", data)
            raise ChatCompletionError(status, data["error"]["message"])
