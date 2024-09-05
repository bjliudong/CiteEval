import os, json
import re
import logging
from openai import OpenAI
import anthropic
import misc
from misc import time_it_s

logger = logging.getLogger(__name__)
data_dir = "data"
openai_baseurl = "https://api.openai.com/v1"
openai_model = "gpt-4o"
'''
gpt-4o
gpt-4o-2024-05-13
gpt-4-turbo
gpt-4-turbo-2024-04-09
gpt-4
gpt-4-32k
gpt-3.5-turbo-0125
gpt-3.5-turbo-instruct
'''
claude_baseurl = "https://api.gptsapi.net"
claude_model = "claude-3-5-sonnet-20240620"
'''
claude-3-opus-20240229
claude-3-sonnet-20240229
claude-3-haiku-20240307
'''
kimi_baseurl = "https://api.moonshot.cn/v1"
kimi_model = "moonshot-v1-128k"
'''
moonshot-v1-8k
moonshot-v1-32k
moonshot-v1-128k
'''
pply_baseurl = "https://api.perplexity.ai"
pply_model = "llama-3-sonar-large-32k-online"
'''
Perplexity Sonar Models
llama-3.1-sonar-small-128k-online   8B
llama-3.1-sonar-large-128k-online   70B
llama-3.1-sonar-huge-128k-online    405B
Perplexity Chat Models
llama-3.1-sonar-small-128k-chat     8B
llama-3.1-sonar-large-128k-chat     70B
Open-Source Models
llama-3.1-8b-instruct               8B
llama-3.1-70b-instruct              70B
'''
models = {
    # openai_model: openai_baseurl,
    # claude_model: claude_baseurl,
    # kimi_model: kimi_baseurl,
    pply_model: pply_baseurl
}
prompt_cn = "说明：仅使用提供的搜索结果（其中一些可能无关紧要）为给定问题写一个准确、引人入胜、简洁的答案，并正确引用。使用公正和新闻的语气。总是引用任何事实主张。当引用多个搜索结果时，请使用[1][2][3]。每句话至少引用一份文件，最多引用三份文件。如果有多个文档支持该句子，则只引用文档中足够小的子集。\n\n"
prompt_en = "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.\n\n"
prompt_gen_cn = "说明：仅使用提供的搜索结果（其中一些可能无关紧要）为给定问题写一个准确、引人入胜、简洁的答案，并正确引用。使用公正和新闻的语气。总是引用任何事实主张。当引用多个搜索结果时，请使用[1][2][3]。每句话至少引用一份文件，最多引用三份文件。如果有多个文档支持该句子，则只引用文档中足够小的子集。\n\n"
prompt_gen_en = "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.\n\n"

@time_it_s
def gen_by_ChatGPT(model, model_baseurl, query_string, conversation_history=[]):
    if 'gpt' in model:
        api_key = os.environ.get("OPENAI_API_KEY")
    elif 'moonshot' in model:
        api_key = os.environ.get("KIMI_API_KEY")
    elif 'llama' in model:
        api_key = os.environ.get("PERPLEXITY_API_KEY")
    else:
        api_key = os.environ.get("WILDCARD_API_KEY")
    baseurl = model_baseurl

    try:
        # 初始化 OpenAI 客户端
        client = OpenAI(
            api_key = api_key,
            base_url = baseurl,
        )
        logger.debug(f"Query: {query_string}")
        # 将历史对话和当前查询合并作为输入
        messages = conversation_history + [
            {"role": "user", "content": f"{query_string}"}
        ]

        # 发送请求到 OpenAI 并获取回复
        completion = client.chat.completions.create(
            model = model,
            messages=messages
        )

        # 获取回复内容
        reply = completion.choices[0].message.content

        # 将当前回复添加到对话历史中
        conversation_history.append(
            {"role": "assistant", "content": reply}
        )

        return reply, conversation_history
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return "", []

@time_it_s
def gen_by_Claude(model_name, model_baseurl, query_string, conversation_history=[]):
    try:
        client = anthropic.Anthropic(
            api_key = os.environ.get("WILDCARD_API_KEY"),
            base_url = model_baseurl,
        )
        messages = conversation_history + [
            {"role": "user", "content": f"{query_string}"}
        ]
        mess = client.messages.create(
            model=model_name,
            max_tokens=1024,
            messages=messages
        )
        reply = mess.content[0].text
        conversation_history.append(
            {"role": "assistant", "content": reply}
        )

        return reply, conversation_history
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return "", []

def gen_answer(query, references, model_name, model_baseurl, lang, answer_history=[]):
    if lang == 'Chinese':
        prompt = prompt_gen_cn
    else:
        prompt = prompt_gen_en

    for index, ref in enumerate(references):
        if index >= 5:
            break
        if lang == 'Chinese':
            prompt = f"{prompt}文档 [{index+1}](标题: {ref['title']}) {ref['summary']} ({ref['url']})\n"
        else:
            prompt = f"{prompt}Document [{index+1}](Title: {ref['title']}) {ref['summary']} ({ref['url']})\n"
    
    if lang == 'Chinese':
        question = f"\n问题: {query}"
    else:
        question = f"\nQuestion: {query}"
    prompt = f"{prompt}\n{question}"
    
    answer = ""
    history = []
    feature = ('gpt', 'moonshot', 'llama')
    if any(_ in model_name for _ in feature):
        answer, history = gen_by_ChatGPT(model_name, model_baseurl, prompt, answer_history)
    elif 'claude' in model_name:
        answer, history = gen_by_Claude(model_name, model_baseurl, prompt, answer_history)
    else:
        logger.error("模型参数错误！")
    return answer, history

# 提取每个语句中的数字引用，并找到其在references中的id，组成特定结构返回，没有引用的话则返回一个空数组
def extract_numbers_and_ref_ids(sentence, references):
    # 使用正则表达式查找所有形如[数字]的模式
    pattern = r'\[\d+\]'
    matches = re.findall(pattern, sentence)
    
    # 提取数字并转换为整数
    numbers = [int(match[1:-1]) for match in matches]
    
    # 根据提取的数字，从references中获取对应的ref_id
    ref_ids = []
    for number in numbers:
        for ref in references:
            if int(ref["idx"]) == number:
                ref_ids.append({"idx": number, "ref_id": ref["ref_id"]})
                break  # 找到匹配的ref_id后跳出循环
    
    return ref_ids

# 将LLM返回的answer逐句拆开，并组成特定结构
def split_answer(answer:str, references:list, lang:str):
    sentences_list = []
    if lang == 'Chinese':
        separator = "。"
    else:
        separator = "."
    sentences = answer.split(separator)
    cleaned_sentences = [sentence.strip('\n ') for sentence in sentences if sentence.strip('\n ') != '']
    for index, sentence in enumerate(cleaned_sentences):
        sentence_dict = {}
        sentence_dict['sentence_id'] = misc.generate_random_code()
        sentence_dict['idx'] = index + 1
        sentence_dict['sentence_string'] = sentence
        sentence_dict['references'] = extract_numbers_and_ref_ids(sentence, references)
        sentences_list.append(sentence_dict)
    return sentences_list

# 处理会话主体内容，将LLM生成的answers加入
def handling_conversations(conversations: list, lang: str):
    llm_answer_history = {key: [] for key in models}
    for conversation in conversations:
        answer_list = []
        query = conversation['query']
        references = conversation['references']
        for model_name, model_baseurl in models.items():
            answer_dict = {}
            answer_dict['answer_id'] = misc.generate_random_code()
            answer, llm_answer_history[model_name] = gen_answer(query, references, model_name, model_baseurl, lang, llm_answer_history[model_name])
            answer_dict['content'] = answer
            answer_dict['model'] = model_name
            answer_dict['sentences'] = split_answer(answer, references, lang)
            answer_list.append(answer_dict)
        conversation['answers'] = answer_list
    return conversations


def read_json_files(directory):    
    for root, _, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否为.json
            if file.endswith('.json'):
                # 构造完整的文件路径
                file_path = os.path.join(root, file)
                # 打开并读取json文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        # 将文件内容转换为字典
                        data = json.load(f)
                        lang = data['conversations']['lang']
                        contents = data['conversations']['contents']
                        data['conversations']['contents'] = handling_conversations(contents, lang)
                        misc.save_json_file(data, f"{file_path}.LLM", "multi", 'w')
                        logger.info(f"保存文件 {file_path}.LLM 成功！")
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON from file {file_path}: {e}")

def main():
    read_json_files(data_dir)

if __name__ == "__main__":
    misc.setup_logging()
    main()