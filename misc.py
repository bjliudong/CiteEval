import os
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