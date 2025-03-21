import os
import re
import asyncio
from dotenv import load_dotenv
from openai import (
    AsyncOpenAI,
    OpenAI,
    AsyncAzureOpenAI,
    AzureOpenAI,
)

# .env 파일 로드
load_dotenv()

# 공통 설정
USE_AZURE = os.getenv("USE_AZURE", "false").lower() == "true"

# 기본 OpenAI 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Azure OpenAI 설정
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini")

# 클라이언트 초기화
if USE_AZURE:
    client = AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    sync_client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
else:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    sync_client = OpenAI(api_key=OPENAI_API_KEY)


# 공통 LLM 호출 함수
def llm_call(prompt: str, model_or_deployment: str = "gpt-4o-mini") -> str:
    messages = [{"role": "user", "content": prompt}]
    chat_completion = sync_client.chat.completions.create(
        model=model_or_deployment if not USE_AZURE else AZURE_DEPLOYMENT_NAME,
        messages=messages,
    )
    return chat_completion.choices[0].message.content


async def llm_call_async(prompt: str, model_or_deployment: str = "gpt-4o-mini") -> str:
    messages = [{"role": "user", "content": prompt}]
    chat_completion = await client.chat.completions.create(
        model=model_or_deployment if not USE_AZURE else AZURE_DEPLOYMENT_NAME,
        messages=messages,
    )
    print(f"{model_or_deployment} 완료")
    return chat_completion.choices[0].message.content


# 테스트 실행
if __name__ == "__main__":
    test = llm_call("안녕")
    print(test)

    # 비동기 테스트 (옵션)
    # asyncio.run(llm_call_async("안녕 async"))
