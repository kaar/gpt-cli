#!/usr/bin/env python3

import argparse
import logging
import os
import sys

import openai

from gpt.completion import ChatCompletion, ChatCompletionRequest, ChatMessage

LOG = logging.getLogger(__name__)

DEFAULT_USER = "gpt-cli"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEFAULT_CHAT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEXT_MODEL = "text-davinci-003"


class ChatClient:
    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        history: list[ChatMessage] = [],
    ):
        self.model = model
        self.messages = history

    def add_system_prompt(self, prompt: str):
        self.messages.append(ChatMessage(role="system", content=prompt))

    def send(self, input_text: str, echo: bool = False):
        """
        Send a message to the chatbot and return the response.

        :param input_text: The message to send to the chatbot.
        :param echo: Whether to echo the input text to the console.
        """

        if echo:
            print(input_text, end="")

        chat_message = ChatMessage(role="user", content=input_text)
        LOG.debug("Sending message: %s", chat_message)
        self.messages.append(chat_message)
        chat_completion = ChatCompletion.create(
            ChatCompletionRequest(
                model=DEFAULT_CHAT_MODEL,
                messages=self.messages,
            )
        )
        LOG.debug("Received message: %s", chat_completion)
        self.messages.append(chat_completion.message)
        return self.messages[-1].content


def main():
    parser = argparse.ArgumentParser(description="OpenAI GPT-3 text completion")
    # If isatty take argument file
    parser.add_argument(
        "file",
        type=argparse.FileType("r"),
        nargs="?",
        default=sys.stdin,
        help="File to read text from",
    )
    parser.add_argument(
        "--instructions",
        "-i",
        type=str,
        help="Instructions to give the model",
    )
    parser.add_argument("--message", "-m", type=str, help="Send message")
    parser.add_argument(
        "--silent",
        "-s",
        action="store_true",
        default=False,
        help="Do not print prompt input for stdin",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="enable debug output",
    )
    args = parser.parse_args()

    messages = []

    if args.debug:
        LOG.setLevel(logging.DEBUG)
        LOG.debug("Verbose output enabled")
        LOG.debug("argv: %s", sys.argv)

    chat = ChatClient(history=messages)
    if args.instructions:
        LOG.debug("Adding instructions: %s", args.instructions)
        chat.add_system_prompt(args.instructions)

    if args.message:
        LOG.debug("Sending message: %s", args.message)
        print(chat.send(args.message))

    if not args.file.isatty():
        LOG.debug("Reading from: %s", args.file.name)
        with args.file as f:
            print(chat.send(f.read(), echo=not args.silent))
            return

    LOG.debug("Starting interactive mode")
    try:
        while True:
            print(chat.send(input(">>> ")))

    except EOFError:
        LOG.debug("EOF")
        return


if __name__ == "__main__":
    try:

        loglevel = logging.INFO
        logging.basicConfig(
            stream=sys.stdout,
            level=loglevel,
            format="%(message)s",
        )
        logging.getLogger("gpt.completion").setLevel(loglevel)
        main()
    except KeyboardInterrupt:
        LOG.debug("KeyboardInterrupt")
        sys.exit(1)
    except openai.OpenAIError as e:
        LOG.error("OpenAIError: %s", e)
