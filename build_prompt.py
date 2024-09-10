import json
import os
import logging
import misc

logger = logging.getLogger(__name__)

CN = "Chinese"
EN = "English"
prompt_doc_en = "Document [{sn}](Title: {Title}) {content} ({url})"
prompt_doc_cn = "文档 [{sn}](标题: {Title}) {content} ({url})"

# 定义一个函数来读取目录下所有的.json文件
def read_json_files(directory):
    data = {}
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            # 构建完整的文件路径
            file_path = os.path.join(directory, filename)
            # 打开并读取文件内容
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    # 将文件内容解析为json格式，并存储在字典中
                    data = json.load(file)
                    process_json_data(data)
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析错误: {file_path} - {str(e)}")
                continue
            except IOError as e:
                logger.error(f"读取文件错误: {file_path} - {str(e)}")
                continue
            except Exception as e:
                logger.error(f"未知错误: {file_path} - {str(e)}")
                continue
    return data

def build_document_prompt(references:list, lang:str):
    prompt = ""
    if lang == CN:
        prompt = prompt_doc_cn
    else:
        prompt = prompt_doc_en
    count = 1
    for reference in references:
        if count > 5:
            break
        prompt += f"{prompt.format(sn=count, Title=reference['Title'], content=reference['content'], url=reference['url'])}\n"
        count += 1
    return prompt

# 定义一个函数来处理读取的json数据
def process_json_data(json_data:dict):
    turn = json_data['conversations']['turn']
    lang = json_data['conversations']['lang']
    topic = json_data['conversations']['topic']
    contents = json_data['conversations']['contents']

def main():
    # 指定目录路径
    directory_path = './data'
    # 调用函数读取.json文件，并处理
    read_json_files(directory_path)

# 主程序
if __name__ == '__main__':
    misc.setup_logging()
    main()