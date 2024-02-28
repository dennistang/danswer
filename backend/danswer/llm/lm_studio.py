import json
from collections.abc import Iterator

import requests
from langchain.schema.language_model import LanguageModelInput
from requests import Timeout

from danswer.configs.model_configs import GEN_AI_API_ENDPOINT
from danswer.configs.model_configs import GEN_AI_MAX_OUTPUT_TOKENS
from danswer.llm.interfaces import LLM
from danswer.llm.utils import convert_lm_input_to_basic_string, convert_lm_input_to_lm_studio_messages
from danswer.utils.logger import setup_logger


logger = setup_logger()


class LMStudioServer(LLM):
    """This class is to provide an example for how to use Danswer
    with any LLM, even servers with custom API definitions.
    To use with your own model server, simply implement the functions
    below to fit your model server expectation

    The implementation below works against the custom FastAPI server from the blog:
    https://medium.com/@yuhongsun96/how-to-augment-llms-with-private-data-29349bd8ae9f
    """

    @property
    def requires_api_key(self) -> bool:
        return False

    def __init__(
        self,
        # Not used here but you probably want a model server that isn't completely open
        api_key: str | None,
        timeout: int,
        endpoint: str | None = GEN_AI_API_ENDPOINT,
        max_output_tokens: int = GEN_AI_MAX_OUTPUT_TOKENS,
        stream: bool = False,
    ):
        if not endpoint:
            raise ValueError(
                "Cannot point Danswer to a custom LLM server without providing the "
                "endpoint for the model server."
            )

        self._endpoint = endpoint
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout
        self._stream = stream

    def _execute(self, input: LanguageModelInput) -> str:
        headers = {
            "Content-Type": "application/json",
        }

        # Take the input, which is a list

        data = {
            "messages": convert_lm_input_to_lm_studio_messages(input),
            "temperature": 0.0,
            "max_tokens": self._max_output_tokens,
            "stream": self._stream
        }
        try:
            response = requests.post(
                self._endpoint, headers=headers, json=data, timeout=self._timeout
            )
        except Timeout as error:
            raise Timeout(f"Model inference to {self._endpoint} timed out") from error

        response.raise_for_status()
        return json.loads(response.content)["choices"][0]["message"]["content"]

    def log_model_configs(self) -> None:
        logger.debug(f"Custom model at: {self._endpoint}")

    def invoke(self, prompt: LanguageModelInput) -> str:
        return self._execute(prompt)

    def stream(self, prompt: LanguageModelInput) -> Iterator[str]:
        yield self._execute(prompt)



        """
          "inputs": "System: You are a helpful assistant.\nDo not provide any citations even if there are examples in the chat history.\n\nAdditional Information:\n\t- The current day and time is Tuesday February 27, 2024 19:35.\nHuman: Introduce yourself.\nHuman: asdfasdfadsf\nHuman: Introduce yourself.",
  "parameters": {
    "temperature": 0,
    "max_tokens": 1024
  }
}
            "messages": [
    {
      "role": "system",
      "content": "Always answer in rhymes."
    },
    {
      "role": "user",
      "content": "Introduce yourself."
    }
  ],
  "temperature": 0.7,
  "max_tokens": -1,
  "stream": false

        """
