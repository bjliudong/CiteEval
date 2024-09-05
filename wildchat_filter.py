import pyarrow.parquet as pq
import numpy as np
import json, string, datetime, random, os, sys
import requests
from serpapi import google_search
from openai import OpenAI
import html2text
import uuid
import pymupdf

# 读取Parquet文件
file_path = 'dataset/train-00002-of-00019.parquet'
output_jsonfile = 'wildchat_filter.json'
src_file = 'wildchat_filter.data'
dict_file = 'wildchat_filter.txt'
temp_file = 'wildchat_filter.tmp'
data_file = "wildchat_filter.data.v2"
serpapi_output = 'serpapi_output'
manually_screened_temp_data_file = 'dataset/train-00002-of-00019_en.txt'
q2k_prompt = "As a search engine expert, please rewrite the following query content as several search keywords, and the total word count of the generated search keywords should not exceed 200 words. The returned keywords should be separated by commas."
q2k_prompt_cn = "你作为搜索引擎专家，请重写下面的查询内容为若干查询关键词，生成的查询关键词总字数不查过200字，返回的关键词以逗号间隔即可。"
LLM_model = "gpt-4o-mini"
temp_pdf_path = "temp_pdf"
PROMPT = "请提取以下html网页的主体内容，将页头、页脚、导航栏以及所有html标签等无效内容过滤掉，注意只输出主体内容，不再输出其他提示信息。\n\n"

# 自定义一个函数来处理不可JSON序列化的对象
def default_serializer(obj):
    """将datetime和np.ndarray对象转换为JSON可序列化的格式"""
    if isinstance(obj, datetime):  # 修正此处，确保是datetime.datetime类型
        return obj.isoformat()  # 将datetime转换为ISO格式的字符串
    elif isinstance(obj, np.ndarray):  # 检查NumPy数组
        return obj.tolist()  # 将NumPy数组转换为列表
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def generate_random_code(length=10):
    # 定义可能的字符集合：大写字母、小写字母和数字
    possible_characters = string.ascii_uppercase + string.ascii_lowercase + string.digits
    
    # 使用 random.choice() 从可能的字符集合中随机选择字符
    random_code = ''.join(random.choice(possible_characters) for _ in range(length))
    
    return random_code

def process_file_path(txt_filepath):
    # 检查文件路径是否有效
    if not os.path.isfile(txt_filepath):
        raise ValueError("The provided file path does not exist or is not a file.")

    # 拆分文件路径
    base_path, filename = os.path.split(txt_filepath)
    name, extension = os.path.splitext(filename)

    # 检查文件名最后的两位字符
    if name[-2:] == "cn":
        lang = "Chinese"
    elif name[-2:] == "en":
        lang = "English"
    else:
        lang = "Unknown"

    # 去掉文件名中的"_cn"或"_en"
    name = name.replace("_cn", "").replace("_en", "")

    # 修改文件扩展名为"parquet"
    new_filename = f"{name}.parquet"

    # 构造新的文件路径
    new_filepath = os.path.join(base_path, new_filename)

    return new_filepath, lang

def read_txt_to_dict(txt_filepath):
    # 初始化一个空字典来存储uid和tid
    uid_tid_dict = {}
    
    # 尝试打开并读取文件
    try:
        with open(txt_filepath, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除每行的首尾空白字符，包括换行符
                line = line.strip()
                # 检查行是否为空
                if not line:
                    continue
                # 以逗号分隔字符串
                parts = line.split(',')
                # 检查分割后是否恰好有两部分
                if len(parts) == 2:
                    uid, tid = parts
                    # 将uid和tid添加到字典中
                    uid_tid_dict[uid] = tid
                else:
                    print(f"Warning: The line '{line}' does not contain exactly one comma and will be skipped.")
    except FileNotFoundError:
        print(f"Error: The file '{txt_filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    # 返回字典
    return uid_tid_dict

def append_dict_to_json_file(data_dict, file_path):
    """
    将字典转换为JSON格式的字符串，并追加到指定文件中。
    如果文件不存在，仅创建该文件。

    参数:
    data_dict (dict): 需要转换为JSON的字典。
    file_path (str): 追加JSON字符串的目标文件路径。
    """
    try:
        # 如果文件不存在，则创建一个空文件
        if not os.path.exists(file_path):
            open(file_path, 'w').close()

        # 将字典转换为JSON格式的字符串
        # json_string = json.dumps(data_dict, ensure_ascii=False, indent=4) + '\n'
        json_string = json.dumps(data_dict, ensure_ascii=False) + '\n'

        # 追加JSON字符串到文件
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(json_string)
    except IOError as e:
        print(f"An IOError occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def check_first_word(input_string):
    # 检查输入是否为null、空字符串或仅包含空格
    if input_string is None or input_string.strip() == "":
        return "no"
    
    # 将输入字符串转换为小写，并按空格分割成单词列表
    words = input_string.lower().split()
    
    # 提取首个单词
    first_word = words[0]
    
    # 定义需要检查的单词列表
    keywords = ['what', 'when', 'why', 'who', 'where', 'how']
    
    # 检查首个单词是否在关键字列表中
    if first_word in keywords:
        return input_string  # 返回原始输入字符串
    else:
        return "no"

def data_process(data):
    # 使用for循环迭代DataFrame中的每一行
    count = 0
    for index, row in data.iterrows():
        # 将DataFrame的行转换为字典
        row_dict = row.to_dict()

        # 获取临时记录
        temp_dict = read_txt_to_dict(manually_screened_temp_data_file)

        lang = row_dict['language']
        turn = row_dict['turn']
        conversation_hash = row_dict['conversation_hash']

        if conversation_hash in temp_dict:
            topic_id = temp_dict[conversation_hash]
            random_code = generate_random_code()
        # if turn < 4 and lang == 'English':
        #     # 首先判断是否以 5W+H 开头，如果不是则跳过本次循环
        #     if check_first_word(row_dict['conversation'][0]['content']) == "no":
        #         continue
            count = count + 1
            json_data = {'id': random_code, 'conversation_hash': conversation_hash}
            conv = {'turn': turn, 'lang':lang, 'topic': topic_id}
            i = 0
            contents = []
            while i < turn*2:
                query = row_dict['conversation'][i]['content']
                answer = row_dict['conversation'][i+1]['content']
                content = {'query': query, 'answer': answer}
                contents.append(content)
                i = i + 2
            conv['contents'] = contents
            json_data['conversations'] = conv
            append_dict_to_json_file(json_data, output_jsonfile)
        else:
            continue

    print("总计取出" + str(count) + "条记录")

def count_turn_and_topic_values(txt_filepath, lang):
    # 初始化计数器
    turn1_count = 0
    turn2_count = 0
    turn3_count = 0
    topic1_count = 0
    topic2_count = 0
    topic3_count = 0
    topic4_count = 0
    topic5_count = 0
    topic6_count = 0
    topic7_count = 0
    topic8_count = 0

    # 以utf8格式逐行读取文件
    with open(txt_filepath, 'r', encoding='utf8') as file:
        for line in file:
            try:
                # 将读取的每行字符串解析为json格式数据，并转换为dict数据结构
                data = json.loads(line.strip())

                if data['conversations']['lang'] == lang:
                    # 根据'turn'的值进行统计
                    turn = data['conversations']['turn']
                    if turn == 1:  # 假设每行都包含'turn'键
                        turn1_count += 1
                    elif turn == 2:
                        turn2_count += 1
                    elif turn == 3:
                        turn3_count += 1

                    # 根据'topic'的值进行统计
                    topic = data['conversations']['topic']  # 假设每行都包含'topic'键
                    if topic == '1':
                        topic1_count += 1
                    elif topic == '2':
                        topic2_count += 1
                    elif topic == '3':
                        topic3_count += 1
                    elif topic == '4':
                        topic4_count += 1
                    elif topic == '5':
                        topic5_count += 1
                    elif topic == '6':
                        topic6_count += 1
                    elif topic == '7':
                        topic7_count += 1
                    elif topic == '8':
                        topic8_count += 1
                else:
                    continue

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from line: {line.strip()} - {e}")
            except KeyError as e:
                print(f"Missing key in JSON data: {e}")

    # 打印统计结果
    print("Counts for 'turn' and 'topic' values:")
    print(f"turn1_count: {turn1_count}")
    print(f"turn2_count: {turn2_count}")
    print(f"turn3_count: {turn3_count}")
    print(f"topic1_count: {topic1_count}")
    print(f"topic2_count: {topic2_count}")
    print(f"topic3_count: {topic3_count}")
    print(f"topic4_count: {topic4_count}")
    print(f"topic5_count: {topic5_count}")
    print(f"topic6_count: {topic6_count}")
    print(f"topic7_count: {topic7_count}")
    print(f"topic8_count: {topic8_count}")

def process_json_lines(src_filepath, dict_filepath, tmp_filepath):
    # 确保输出文件存在，如果不存在则创建
    if not os.path.exists(dict_filepath):
        open(dict_filepath, 'a').close()
    if not os.path.exists(tmp_filepath):
        open(tmp_filepath, 'a').close()

    # 打开源文件并逐行读取
    with open(src_filepath, 'r', encoding='utf-8') as src_file:
        for line in src_file:
            try:
                # 解析JSON格式的字符串
                row_dict = json.loads(line)
                
                # 检查并修改 'conversations' 下的 'topic' 值
                if 'conversations' in row_dict and 'topic' in row_dict['conversations']:
                    if row_dict['conversations']['topic'] == "3":
                        row_dict['conversations']['topic'] = "4"  # 修改topic为4
                
                # 根据 'topic' 的值写入不同的文件
                if row_dict['conversations']['topic'] == "2":
                    append_dict_to_json_file(row_dict, temp_file)
                else:
                    append_dict_to_json_file(row_dict, dict_file)

            except json.JSONDecodeError:
                print(f"Warning: Failed to decode JSON from line: {line}")
            except KeyError as e:
                print(f"Warning: Missing key in JSON data - {e}")

def merge_and_save_files(data_filepath, f1_filepath, f2_filepath):
    # 读取文件内容
    try:
        with open(f1_filepath, 'r', encoding='utf-8') as file1:
            content_f1 = file1.read()
        
        with open(f2_filepath, 'r', encoding='utf-8') as file2:
            content_f2 = file2.read()
        
        # 合并内容
        merged_content = content_f1 + content_f2
        
        # 将合并后的内容写入data_filepath
        with open(data_filepath, 'w', encoding='utf-8') as data_file:
            data_file.write(merged_content)
            
    except FileNotFoundError as e:
        print(f"Error: One of the files does not exist - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def query_to_keyword(query_string):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    completion = client.chat.completions.create(
        model = LLM_model,
        messages=[
            {"role": "user", "content": f"{query_string}"}
        ]
    )
    return completion.choices[0].message.content

def get_organic_results(json_str):
    try:
        # 将输入的字符串解析为字典
        data_dict = json.loads(json_str)
        
        # 检查是否存在 'organic_results' 键
        if 'organic_results' in data_dict:
            # 迭代 'organic_results' 数组
            for item in data_dict['organic_results']:
                # 检查当前字典是否有 'link' 键
                if 'link' in item:
                    # 提取 'link' 的值
                    link = item['link']
                    # 判断链接的文件扩展名
                    if link.lower().endswith('.pdf'):
                        res = fetch_pdf_text(download_pdf(link, temp_pdf_path))
                    else:
                        res = content_filter(fetch_url_content(link))
            return res
        else:
            print("No 'organic_results' key found in the JSON data.")
            return ""
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return ""

def search_and_save(params, num, filename):
    search = google_search.GoogleSearch(params)
    result = search.get_dict()
    result = {num: result}
    append_dict_to_json_file(result, filename)

def serpapi_search(data_filepath, output_filepath):
    # 确保输出目录存在
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    # 计数器初始化
    count = 0
    # 打开输入文件进行读取
    with open(data_filepath, 'r', encoding='utf-8') as file:
        for line in file:
            count += 1
            if count > 20:
                break
            # 将读取到的每一行解析为JSON格式
            row_dict = json.loads(line)

            params = {
                "num": "100",
                "api_key": os.environ.get("SERPAPI_KEY"), 
                "output": "json"
            }
            lang = row_dict['conversations']['lang']
            if lang == 'English':
                params['hl'] = 'en'
                params['gl'] = 'us'
                query = "query:"
                answer = "answer:"
                query_string = q2k_prompt
            elif lang == 'Chinese':
                params['hl'] = 'zh-cn'
                params['gl'] = 'cn'
                query = "问："
                answer = "答："
                query_string = q2k_prompt_cn

            # 构建输出文件的文件名
            filename = f"{output_filepath}/{row_dict['conversation_hash']}.data"
            contents = row_dict['conversations']['contents']
            # 遍历列表中的每个字典
            for i in range(len(contents)):
                # 初始化一个空字符串用于存放当前的输出结果
                if lang == 'English':
                    query_string = q2k_prompt
                elif lang == 'Chinese':
                    query_string = q2k_prompt_cn
                
                # 对于当前索引i，输出从索引0到i的所有字典中的键值对
                for j in range(i + 1):
                    # 遍历每个字典中的键值对
                    for key, value in contents[j].items():
                        # 根据键的名称来决定输出的前缀
                        if "query" in key:
                            query_string += f"{query} {value}\n"
                        elif "answer" in key:
                            if j < i:
                                query_string += f"{answer} {value}\n"
                
                keyword = query_to_keyword(query_string)
                params['q'] = keyword
                # print(f"===keyword: {keyword}\n")
                search_and_save(params, i+1, filename)

def content_filter_LLM(content):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    completion = client.chat.completions.create(
        model = LLM_model,
        messages=[
            {"role": "user", "content": f"{PROMPT}{content}"}
        ]
    )
    return completion.choices[0].message.content

def content_filter(html_content):
    # 创建一个 HTML 到 Markdown 的转换器实例
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_mailto_links= True
    h.ignore_tables = True
    h.ignore_emphasis = True
    # 使用 html2text 转换 HTML 到 Markdown
    markdown_content = h.handle(html_content)
    return markdown_content

def fetch_url_content(url):
    try:
        # 发送GET请求
        response = requests.get(url)
        # 尝试从HTTP头信息中获取编码
        encoding = response.apparent_encoding
        # 确保请求成功
        response.raise_for_status()
        # 根据编码解码网页内容
        content = response.content.decode(encoding)
        return content_filter(content)
    except requests.RequestException as e:
        # 打印错误信息
        print(f"An error occurred: {e}")
        return None

def download_pdf(url, directory):
    """
    下载PDF文件并保存到指定目录，文件名为随机UUID。
    
    :param url: PDF文件的URL
    :param directory: 保存PDF文件的目录路径
    """
    # 确保目录存在
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 生成随机的UUID作为文件名
    file_name = str(uuid.uuid4()) + '.pdf'
    
    # 完整的文件路径
    file_path = os.path.join(directory, file_name)
    
    # 发送HTTP GET请求
    response = requests.get(url, stream=True)
    
    # 检查请求是否成功
    if response.status_code == 200:
        # 打开文件进行写入
        with open(file_path, 'wb') as file:
            # 写入内容到文件
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        # print(f"PDF文件已下载并保存到：{file_path}")
    else:
        print(f"下载失败，状态码：{response.status_code}")
    return file_path

def fetch_pdf_text(pdf_filepath):
    with pymupdf.open(pdf_filepath) as doc:  # open document
        text = chr(12).join([page.get_text() for page in doc])
    return text

def main():
    try:
        parquet_file = pq.ParquetFile(file_path)
        table = parquet_file.read()
        data = table.to_pandas()
        data_process(data)
    except Exception as e:
        print(f"An error occurred: {e}")

def static(lang):
    count_turn_and_topic_values(output_jsonfile, lang)

def merge():
    process_json_lines(src_file, dict_file, temp_file)

def merge_save():
    merge_and_save_files(data_file, dict_file, temp_file)

def search_ref():
    serpapi_search(src_file, serpapi_output)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "static_cn":
            static('Chinese')
        elif sys.argv[1] == "static_en":
            static('English')
        elif sys.argv[1] == "main":
            main()
        elif sys.argv[1] == "merge":
            merge()
        elif sys.argv[1] == "merge_save":
            merge_save()
        elif sys.argv[1] == "search_ref":
            search_ref()
        else:
            print("Invalid command.")
    else:
        main()