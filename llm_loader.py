from langchain.chat_models import ChatOpenAI

def load_llama_model():
    return ChatOpenAI(

        #Llama 3 8B 모델로 RAG 실행하고 싶은 경우
        #base_url="http://torch27:8000/v1",
        #model="meta-llama/Meta-Llama-3-8B-Instruct",
        #openai_api_key="EMPTY"

        #Exaone으로 RAG 실행하고 싶은 경우
        base_url="http://220.124.155.35:8000/v1",
        model="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        openai_api_key="token-abc123"

        #base_url="https://7xiebe4unotxnp-8000.proxy.runpod.net/v1",
        #model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        #openai_api_key="EMPTY"

       # base_url="http://vllm_yjy:8000/v1",
       # model="/models/Llama-3.3-70B-Instruct-AWQ",
       # openai_api_key="token-abc123"

    )
