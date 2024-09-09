import json
import os
import logging
import misc

logger = logging.getLogger(__name__)

def is_zero_byte_json(file_path):
    if os.path.getsize(file_path) == 0:
        return True
    else:
        return False


def read_json_file(file_path):
    """以UTF-8格式，以只读模式读取JSON文件，并解析为Python的dict数据结构"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        logger.info(f"成功读取并解析JSON文件: {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {file_path} - {str(e)}")
        return None
    except IOError as e:
        logger.error(f"读取文件错误: {file_path} - {str(e)}")
        return None
    except Exception as e:
        logger.error(f"未知错误: {file_path} - {str(e)}")
        return None

def write_json_file(file_path, data_dict):
    """以UTF-8格式，以只写模式写入JSON文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data_dict, file, ensure_ascii=False, indent=4)
        logger.info(f"成功写入JSON文件: {file_path}")
    except IOError as e:
        logger.error(f"写入文件错误: {file_path} - {str(e)}")
    except Exception as e:
        logger.error(f"写入JSON文件时发生未知错误: {file_path} - {str(e)}")


def process_contents_from_dict(data_dict):
    """处理contents数据项"""
    count = 0
    try:
        contents = []
        contents = data_dict.get('conversations', {}).get('contents', [])
        contents, count = process_references_from_contents(contents)
        data_dict.get('conversations', {}).update({'contents': contents})
        return data_dict, count
    except Exception as e:
        logger.error(f"处理contents数据时发生错误: {str(e)}")
        return data_dict, count  # 返回原始数据字典，以防止数据丢失


def process_references_from_contents(contents):
    """处理contents中的references数据项"""
    ret = []
    count = 0
    for content in contents:
        try:
            content['references'], count = del_empty_summary_from_references(content.get('references', []))
            ret.append(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {str(e)}")
        except KeyError as e:
            logger.error(f"访问字典键时出错: {str(e)}")
        except Exception as e:
            logger.error(f"处理content时发生未知错误: {str(e)}")
    return ret, count


def del_empty_summary_from_references(references):
    """删除summary为空或为"不相关"的references数据项"""
    ret = []
    count = 0
    for ref in references:
        try:
            summ = ref.get('summary', '')
            if summ not in ["不相关", "irrelevant", ""]:
                ret.append(ref)
            else:
                count += 1
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {str(e)}")
            continue
        except KeyError as e:
            logger.error(f"访问字典键时出错: {str(e)}")
            continue
        except Exception as e:
            logger.error(f"处理reference时发生未知错误: {str(e)}")
            continue
    return ret, count

def process_json_files(directory):
    """遍历指定目录下的所有JSON文件，并执行上述功能"""
    total_count = 0
    empty_count = 0
    process_count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            if is_zero_byte_json(file_path):
                logger.info(f"Zero byte file found: {file_path}")
                empty_count += 1
                continue
            count = 0
            data_dict = read_json_file(file_path)
            data_dict, count = process_contents_from_dict(data_dict)
            write_json_file(file_path, data_dict)
            logger.info(f"序号{total_count+1}，处理完成{filename}，删除的无效reference数量: {count}")
            if count > 0:
                process_count += 1
            total_count += 1
    logger.info(f"处理完成，总处理数量：{total_count}，空文件数量: {empty_count}，有效处理文件数量: {process_count}")

if __name__ == "__main__":
    misc.setup_logging()
    process_json_files('./data')