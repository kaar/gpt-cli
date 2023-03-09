import pytest
from datetime import datetime

from gpt.completion import ChatCompletionResponse


def test_completion_response():
    json = {
        "id": "chatcmpl-6p9XYPYSTTRi0xEviKjjilqrWU2Ve",
        "object": "chat.completion",
        "created": 1677649420,
        "model": "gpt-3.5-turbo",
        "usage": {"prompt_tokens": 56, "completion_tokens": 31, "total_tokens": 87},
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Some text",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }

    response = ChatCompletionResponse(**json)

    assert response.id == "chatcmpl-6p9XYPYSTTRi0xEviKjjilqrWU2Ve"
    assert response.object == "chat.completion"
    assert response.created == datetime.fromtimestamp(1677649420)
    assert response.model == "gpt-3.5-turbo"
    assert response.usage.prompt_tokens == 56
    assert response.usage.completion_tokens == 31
    assert response.usage.total_tokens == 87
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "Some text"
    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].index == 0
