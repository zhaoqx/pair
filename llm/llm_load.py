def get_api_config(api_name):
    # 读取key.csv文件获取api_name对应的API密钥
    import pandas as pd
    df = pd.read_csv('configs/key.csv')
    # 使用loc方法根据api查api_key
    api_key = df.loc[df['api'] == api_name, 'api_key'].values[0]
    api_url = df.loc[df['api'] == api_name, 'api_url'].values[0]
    return api_key, api_url

def langchain_chatopenai_llm(llm_name):
    from langchain_openai import ChatOpenAI
    import os
    match llm_name:
        case ('qwen-max' | 'qwq-32b-preview'):
            api_key, api_url = get_api_config('qwen')
        case ('glm-4-flash'):
            api_key, api_url = get_api_config('bigmodel')
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_url
    return ChatOpenAI(model_name=llm_name, base_url=api_url)
            
def langchain_chatmistral_llm(llm_name):
    from langchain_mistralai import ChatMistralAI
    import os
    match llm_name:
        case ('mistral-large-latest'):
            api_key, api_url = get_api_config('mistral')
    os.environ["MISTRAL_API_KEY"] = api_key
    return ChatMistralAI(model=llm_name)

def langchain_chatollama_llm(llm_name):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=llm_name)

def choose_llm(type, llm_name):
    # 基本的api-key
    match type:
        case 'langchain_chatopenai':
            return langchain_chatopenai_llm(llm_name)
        case 'langchain_chatmistral':
            return langchain_chatmistral_llm(llm_name)
        case 'langchain_chatollama':
            return langchain_chatollama_llm(llm_name)
        