import os, json, time
import logging
from serpapi import google_search
import html2text
import pymupdf, uuid
import requests
from pathlib import Path
import misc
from misc import time_it
from misc import time_it_s
import ollama
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

data_file = "wildchat_filter.data"
temp_dir = "temp"
data_dir = "data"
LLM_model = "qwen2:7b"
prompt_q2k_en = "As a search engine expert, please rewrite the following query content as several search keywords, and the total word count of the generated search keywords should not exceed 200 words. The returned keywords should be separated by commas."
prompt_q2k_cn = "你作为搜索引擎专家，请重写下面的查询内容为若干查询关键词，生成的查询关键词总字数不查过200字，返回的关键词以逗号间隔即可。"
prompt_summ_cn = "用感兴趣的问题在100个字内总结以下正文。如果文档与问题无关，请返回“不相关”。尽量保留所有重要的日期、数字和姓名。\n\n"
prompt_summ_en = "Summarize the following document within 50 words with the question of interest Return \"irrelevant\" if the document is irrelevant to the question. Try to keep all the important dates, numbers, and names.\n\n"
logger = logging.getLogger(__name__)

# 调用 LLM 将输入的问答对，转换为查询关键词
@time_it_s
def llm_answer(query_string):
    try:
        client = ollama.Client(host=os.environ.get('OLLAMA_HTTP'))
        response = ollama.chat(model=LLM_model, messages=[
        {
            'role': 'user',
            'content': query_string,
        },
        ])
        return response['message']['content']
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None

# 利用 html2text 获取网页内容
def get_html_content(html_content):
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

# 利用 requests 抓取指定url的网页内容
@time_it
def get_webpage_content(url):
    logger.info(f"开始抓取网页，{url}")
    try:
        response = requests.get(url, timeout=(5, 20)) # 设置连接超时5秒，读取超时20秒
        # 尝试从HTTP头信息中获取编码
        encoding = response.apparent_encoding or 'utf-8'
        # 确保请求成功
        response.raise_for_status()
        # 根据编码解码网页内容
        content = response.content.decode(encoding)
        # logger.info(f"抓取网页成功！ 地址：{url}")
        return get_html_content(content)
    except requests.exceptions.Timeout as e:
        logger.error(f"抓取网页连接超时: {e}")
        return None
    except requests.RequestException as e:
        logger.error(f"An error occurred: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None

# 抓取指定网址的pdf文件
@time_it
def get_pdf_content(url, conversation_hash, timeout=5, max_download_time=300):
    logger.info(f"开始抓取pdf，{url}")
    try:
        # 生成随机的UUID作为文件名
        file_name = str(uuid.uuid4()) + '.pdf'
        file_path = os.path.join(f"{temp_dir}/{conversation_hash}", file_name)
        
        # 发送HTTP GET请求
        response = requests.get(url, stream=True, timeout=timeout)
        
        # 检查请求是否成功
        if response.status_code == 200:
            # 获取文件总大小
            total_size = int(response.headers.get('content-length', 0))
            if total_size == 0:
                logger.warning("无法获取文件大小，进度条可能无法正确显示。")
            
            start_time = time.time()
            # 打开文件进行写入，并显示进度条
            with open(file_path, 'wb') as file, tqdm(
                desc=file_name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                # 写入内容到文件
                for chunk in response.iter_content(chunk_size=8192):
                    if time.time() - start_time > max_download_time:
                        logger.error(f"下载时间超过{max_download_time}秒，强制中断下载。")
                        return None
                    size = file.write(chunk)
                    progress_bar.update(size)
            
            # 打印下载成功的消息
            logger.info(f"PDF文件已下载并保存到：{file_path}")
        else:
            logger.info(f"下载失败，状态码：{response.status_code}, 地址：{url}")
            return None  # 返回None表示下载失败
    except requests.exceptions.Timeout as e:
        logger.error(f"抓取PDF连接超时: {e}")
        return None
    except requests.RequestException as e:
        logger.error(f"请求URL时发生错误：{e}, 地址：{url}")
        return None  # 返回None表示请求失败
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}, 地址：{url}")
        return None

    try:
        # 尝试打开PDF文件并读取文本
        with pymupdf.open(file_path) as doc:
            text = '\n'.join([page.get_text() for page in doc])
        # logger.info(f"抓取：{file_path}文件内容成功！")
        return text
    except Exception as e:
        logger.error(f"打开或读取PDF文件时发生错误：{e}")
        return None

def gen_summ(question, title, text, lang):
    if lang == 'Chinese':
        prompt = f"{prompt_summ_cn}问题: {question}\n标题: {title}\n正文: {text}\n摘要: "
    else:
        prompt = f"{prompt_summ_en}Question: {question}\nTitle: {title}\nText: {text}\nSummary: "
    return llm_answer(prompt)

# 根据输入的 serpapi 搜索到的 json 数据，抽取有用的内容补充到原有字典中，并返回更新后的dict
def supplement_ref(serpapi_json: dict, content: dict, lang, conversation_hash):
    count = 0
    if serpapi_json['search_information']['total_results'] == 0:
        logger.info(f"搜索结果为空，忽略本条记录！！")
        return content
    organic_results = serpapi_json['organic_results']
    references = []
    for index, item in enumerate(organic_results):
        ref = {}
        ref['ref_id'] = misc.generate_random_code()
        ref['idx'] = index
        ref['index'] = item['position']
        ref['title'] = item['title']
        try:
            ref['snippet'] = item['snippet']
        except KeyError:
            ref['snippet'] = ""  # 如果没有snippet，设置为空字符串
        link = item['link']
        ref['url'] = link
        url = str(link).lower()
        if url.endswith('pdf'):
            ref['type'] = 'PDF'
            c = get_pdf_content(link, conversation_hash)
        elif url.endswith('txt'):
            ref['type'] = 'Text'
            c = get_webpage_content(link)
        elif url.endswith('md'):
            ref['type'] = 'Markdown'
            c = get_webpage_content(link)
        else:
            ref['type'] = 'WebPage'
            c = get_webpage_content(link)
        if c is None:
            logger.info("无法抓取url地址内容，忽略本条记录！！")
            continue
        else:
            ref['main_body'] = c
            if count < 10:
                summ = gen_summ(content['query'], item['title'], c, lang)
                ref['summary'] = summ
            else:
                ref['summary'] = ""
            if summ is not None:
                count += 1
            
        references.append(ref)
    content['references'] = references
    return content

# 调用搜索引擎，将搜索到的ref数据补充到原有的json数据中
def build_json(json_data: dict):
    lang = json_data['conversations']['lang']
    contents = json_data['conversations']['contents']
    new_contents = []
    query_keyword_file = f"{temp_dir}/{json_data['conversation_hash']}/{json_data['conversation_hash']}_keyword.data"
    search_result_file = f"{temp_dir}/{json_data['conversation_hash']}/{json_data['conversation_hash']}_search.data"
    params = {
        "num": "100",
        "api_key": os.environ.get("SERPAPI_KEY"), 
        "output": "json"
    }
    if lang == 'English':
        params['hl'] = 'en'
        params['gl'] = 'us'
        query = "query:"
        answer = "answer:"
        query_string = prompt_q2k_en
    elif lang == 'Chinese':
        params['hl'] = 'zh-cn'
        params['gl'] = 'cn'
        query = "问："
        answer = "答："
        query_string = prompt_q2k_cn
    
    # 遍历列表中的每个字典
    for i in range(len(contents)):
        if lang == 'English':
            query_string = prompt_q2k_en
        elif lang == 'Chinese':
            query_string = prompt_q2k_cn
        
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
        
        # 利用 LLM 将问答对转换为搜索关键词
        keyword = llm_answer(query_string)
        # 将关键词保存到文件中
        misc.save_json_file(keyword, query_keyword_file, 'single', 'w')
        logger.info(f"保存搜索关键词到{query_keyword_file}文件成功！")

        params['q'] = keyword
        # 检查搜索结果文件是否已存在，如果存在则从文件中读取搜索结果，避免重复调用 SerpApi 进行检索
        if os.path.exists(search_result_file):
            # 如果文件已存在，从文件中读取搜索结果
            with open(search_result_file, 'r', encoding='utf-8') as file:
                search_result = json.load(file)
            logger.info(f"从文件{search_result_file}中读取已有搜索结果")
        else:
            # 如果文件不存在，调用 SerpApi 进行检索
            search = google_search.GoogleSearch(params)
            search_result = search.get_dict()
            # 将检索到的结果保存下来
            misc.save_json_file(search_result, search_result_file, 'single', 'w')
            logger.info(f"保存搜索结果到{search_result_file}文件成功！")
        
        # 将搜索到的ref数据补充到原有的json数据中
        new_contents.append(supplement_ref(search_result, contents[i], lang, json_data['conversation_hash']))

    json_data['conversations']['contents'] = new_contents
    return json_data

def process_single_record(line, count, temp_dir, data_dir):
    row_dict = json.loads(line)
    # 如果本条记录已经处理过则跳过
    if(misc.check_file_exists(f"{data_dir}/{row_dict['conversation_hash']}.json")):
        return
    logger.info(f"开始处理，序号：{count}，id: {row_dict['id']}，hash: {row_dict['conversation_hash']}")
    # 确保临时目录存在
    if not os.path.exists(f"{temp_dir}/{row_dict['conversation_hash']}"):
        os.makedirs(f"{temp_dir}/{row_dict['conversation_hash']}")
    json_dict = build_json(row_dict)
    json_filename = f"{temp_dir}/{row_dict['conversation_hash']}/{row_dict['conversation_hash']}.json"
    misc.save_json_file(json_dict, json_filename, "multi")
    logger.info(f"保存文件{json_filename}成功！")
    # 将刚才处理过的所有文件从临时目录移动到目标数据目录，并删除临时目录
    misc.move_files(f"{temp_dir}/{row_dict['conversation_hash']}", data_dir)
    os.rmdir(f"{temp_dir}/{row_dict['conversation_hash']}")
    logger.info(f"处理完成，序号：{count}，id: {row_dict['id']}，hash: {row_dict['conversation_hash']}")

def main():
    # 确保输出目录存在
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # 使用普通计数器和锁
    count = 0
    count_lock = threading.Lock()
    
    # 打开输入文件进行读取
    with open(data_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    def process_with_count(line):
        nonlocal count
        with count_lock:
            count += 1
            current_count = count
        process_single_record(line, current_count, temp_dir, data_dir)

    # 创建一个线程池
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 提交所有任务到线程池
        list(tqdm(executor.map(process_with_count, lines), total=len(lines)))

    logger.info(f"总共处理了 {count} 条记录")

if __name__ == "__main__":
    misc.setup_logging()
    main()