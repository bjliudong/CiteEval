import os
from pathlib import Path
import json
import string
import random
import timeit
from functools import wraps
import logging
import logging.config
import yaml
import inspect
import tiktoken
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests

logger = logging.getLogger(__name__)

def setup_logging(default_path='logging_config.yaml'):
    path = os.path.join(os.path.dirname(__file__), default_path)
    if os.path.exists(path):
        with open(path, 'rt', encoding='utf-8') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        print("Logging configuration file not found: using default logging.")

# 检查指定文件是否存在
def check_file_exists(file_path):
    return os.path.isfile(file_path)

# 程序运行计时器
def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 获取函数的签名
        sig = inspect.signature(func)
        # 构造参数列表
        params = sig.bind(*args, **kwargs).arguments
        param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
        
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        
        # 打印函数名和参数
        logger.info(f"{func.__name__}({param_str}) took {(end - start) * 1000:.3f} ms to execute")
        
        return result
    return wrapper

# 程序运行计时器(简化版)
def time_it_s(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()  # 开始计时
        result = func(*args, **kwargs)  # 执行函数
        end = timeit.default_timer()    # 结束计时
        logger.info(f"{func.__name__} took {(end - start) * 1000:.3f} ms to execute")
        return result
    return wrapper

# 生成随机码
def generate_random_code(length=10):
    # 定义可能的字符集合：大写字母、小写字母和数字
    possible_characters = string.ascii_uppercase + string.ascii_lowercase + string.digits
    
    # 使用 random.choice() 从可能的字符集合中随机选择字符
    random_code = ''.join(random.choice(possible_characters) for _ in range(length))
    
    return random_code

# 将输入的 dict 数据保存到 json 文件中
def save_json_file(data_dict, file_path, mode='single', file_mode='a'):
    try:
        # 如果文件不存在，则创建一个空文件
        if not os.path.exists(file_path):
            open(file_path, 'w').close()

        # 将字典转换为JSON格式的字符串
        if mode == 'single':
            json_string = json.dumps(data_dict, ensure_ascii=False) + '\n'
        else:
            json_string = json.dumps(data_dict, ensure_ascii=False, indent=4) + '\n'

        # 追加JSON字符串到文件
        with open(file_path, file_mode, encoding='utf-8') as file:
            file.write(json_string)
    except IOError as e:
        logger.error(f"An IOError occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

# 使用OpenAI的tokenizer计算token数量
def calculate_token_count(text, logger, model='gpt-3.5-turbo'):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    token_count = len(tokens)
    word_count = len(text)
    logger.info(f"Token count: {token_count}\nOriginal Text Count: {word_count}")
    return token_count

# 下载文件
def download_file(url, num_threads=5, filename='download.pdf', is_single=False, chunk_size=1024*1024):
    try:
        file_size = get_file_size(url)
        if not is_single and file_size is not None:
            download_pdf_multi_thread(url, num_threads, filename, file_size)
        else:
            download_pdf_chunked(url, num_threads, filename, chunk_size)
    except Exception as e:
        logger.info(f"发生错误: {e}")
        logger.info("将使用单线程下载方式")
        download_file_single_thread(url, filename)


# 定义一个下载PDF部分的函数
def download_chunk(url, start, end, filename):
    headers = {'Range': f'bytes={start}-{end}'}
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'r+b') as f:
            f.seek(start)
            f.write(r.content)

def download_pdf_chunked(url, num_threads, filename, chunk_size):
    with open(filename, 'wb') as f:
        pass  # 创建空文件

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        offset = 0
        while True:
            futures = []
            for _ in range(num_threads):
                future = executor.submit(download_chunk_no_size, url, offset, offset + chunk_size - 1, filename)
                futures.append(future)
                offset += chunk_size

            completed = False
            for future in futures:
                result = future.result()
                if not result:
                    completed = True
                    break

            if completed:
                break

def download_chunk_no_size(url, start, end, filename):
    headers = {'Range': f'bytes={start}-{end}'}
    with requests.get(url, headers=headers, stream=True) as r:
        if r.status_code == 416:  # 请求范围不满足
            return False
        r.raise_for_status()
        with open(filename, 'r+b') as f:
            f.seek(start)
            f.write(r.content)
    return True

# 计算文件总大小
def get_file_size(url):
    try:
        response = requests.head(url)
        response.raise_for_status()
        return int(response.headers.get('Content-Length', 0))
    except:
        return None

# 多线程下载函数
@time_it_s
def download_pdf_multi_thread(url, num_threads, filename, file_size):
    part_size = file_size // num_threads
    # 创建一个空的文件
    with open(filename, 'wb') as f:
        f.truncate(file_size)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename) as pbar:
            for i in range(num_threads):
                start = i * part_size
                end = (i + 1) * part_size - 1 if i < num_threads - 1 else file_size - 1
                future = executor.submit(download_chunk_with_progress, url, start, end, filename, pbar)
                futures.append(future)
            
            for future in futures:
                future.result()  # 等待每个线程完成

def download_chunk_with_progress(url, start, end, filename, pbar):
    headers = {'Range': f'bytes={start}-{end}'}
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'r+b') as f:
            f.seek(start)
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return True


# 单线程下载文件的函数
@time_it_s
def download_file_single_thread(url, filename='output.pdf'):
    try:
        # 发送GET请求获取文件内容
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 如果请求不成功则抛出异常
        
        # 获取文件总大小
        file_size = int(response.headers.get('Content-Length', 0))
        if file_size == 0:
            logger.warning("警告：无法获取文件大小，进度条可能无法正确显示。")
        
        # 打开文件准备写入
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=file_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
        
        logger.info(f"文件 {filename} 下载完成")
    except requests.RequestException as e:
        logger.info(f"下载失败: {e}")
    except IOError as e:
        logger.info(f"写入文件时出错: {e}")

# 将文件从临时目录移动到目标数据目录
@time_it
def move_files(temp_dir, data_dir):
    # 创建Path对象
    temp_path = Path(temp_dir)
    data_path = Path(data_dir)

    # 确保目标目录存在
    data_path.mkdir(parents=True, exist_ok=True)

    # 遍历源目录中的所有文件并移动
    for file_path in temp_path.iterdir():
        if file_path.is_file():  # 确保是文件
            destination_path = data_path / file_path.name
            file_path.replace(destination_path)  # 移动文件
            logger.debug(f"文件 '{file_path.name}' 已移动到 '{data_path}'。")