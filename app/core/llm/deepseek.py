from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
import requests


class ChatDeepSeekOpenRouter(BaseChatModel):
    """OpenRouter DeepSeek 기반 커스텀 LLM"""

    model_name: str = Field(alias="model")
    openai_api_key: str
    openai_api_base: str = "https://openrouter.ai/api/v1"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    timeout: Optional[int] = 60

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = [{"role": "user", "content": messages[-1].content}]
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = requests.post(
            f"{self.openai_api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        message = AIMessage(
            content=content,
            response_metadata={"model_name": self.model_name},
            usage_metadata={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "deepseek-openrouter"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "openai_api_base": self.openai_api_base,
        }
