from llm.llm_load import choose_llm

def simple_translate_chat(llm, source_word):
    from langchain_core.prompts import ChatPromptTemplate

    system_message = "请对下列单词或句子进行英汉互译，将英语翻译成汉语，将汉语翻译成英语"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_message), ("user", "{text}")]
    )

    prompt = prompt_template.invoke({"text": source_word})
    response = llm.invoke(prompt)
    return response

# 验证chat
llm = choose_llm('langchain_chatmistral', 'mistral-large-latest')
# llm = choose_llm('langchain_chatopenai', 'qwen-max')
# llm = choose_llm('langchain_chatollama', 'llama3.1')
response = simple_translate_chat(llm, "研发过程质量考核 interface")
print(response.content)