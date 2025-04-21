from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv

from core.llm.deepseek import ChatDeepSeekOpenRouter

load_dotenv()

# 커스텀 LLM 인스턴스 생성
custom_llm = ChatDeepSeekOpenRouter(
    model="deepseek/deepseek-chat-v3-0324:free",
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# 메시지 생성 및 응답 받기
messages = [HumanMessage(content="인생의 의미란 무엇일까? 한국어로 대답해")]
response = custom_llm._generate(messages)

# 응답 출력
print(response.generations[0].message.content)
