# CiteEval

Citation evaluation

## Installation and deployment

Create a Python virtual environment

```shell
python -m venv myenv

source myenv/bin/activate

pip install -r requirements.txt
```

## Code Description

`wildchat_filter.py` This code is responsible for obtaining Chinese or English conversation data from WildChat, while filtering conversation with less than or equal 3 turns. Finally, combining the topic classification marked manually for the conversation, generate `wildhat_filter.data`.

`build_ref.py` Read wildchat_filter.data, convert the conversation into keywords, and call serpapi to search for reference citations. This process will save the keywords and search result list converted from the LLM to a temporary file. If the search results contain PDF files, they will also be downloaded to the same temporary directory. Call the LLM to perform text summarization on the obtained reference references and save the final result to a JSON file.

`llm_gen_answer.py` This code processes the results generated by `build_def.cy`. Retrieve Q&A pairs and reference citation content from JSON files, and call LLM to generate several sets of answers with citations. At present, OpenAI ChatGPT, Moonshot kimi, Perpexity and Claude business models have been called. In the future, open source models such as Llama and Qwen are planned to be supplemented.

## Data Sample

The JSON data contained in `wildchat_filter.data`

#### Chinese

```json
{
  "id": "TkY7cRpjSg",
  "conversation_hash": "13e322fb353e1c37ed9a54e526135486",
  "conversations": {
    "turn": 3,
    "lang": "Chinese",
    "topic": "6",
    "contents": [
      {
        "query": "国庆节各个国家放假多少天，其中有哪些国家是需要调休的",
        "answer": "以下是一些国家国庆节放假的情况：\n\n1. 中国：国庆节放假7天，从10月1日至10月7日。\n\n2. 韩国：国庆节放假3天，从10月3日至10月5日。\n\n3. 日本：国庆节放假3天，从10月1日至10月3日。\n\n4. 新加坡：国庆节放假1天，10月1日。\n\n5. 美国：国庆节不是全国性的法定假日，但通常是放假的，放假时间依据各州和企业的规定。\n\n有些国家为了保证工作效率，需要将国庆节放假期间的工作日进行调休。中国是一个例子，国庆节前的周末和国庆节后的周末会各增加一天假期来进行调整。"
      },
      {
        "query": "劳动节节各个国家放假多少天，其中有哪些国家是需要调休的",
        "answer": "以下是一些国家劳动节放假的情况：\n\n1. 中国：劳动节放假3天，从5月1日至5月3日。\n\n2. 美国：劳动节是全国性的法定假日，通常放假1天，即劳动节当天（9月的第一个星期一）。\n\n3. 加拿大：劳动节是全国性的法定假日，通常放假1天，即劳动节当天（9月的第一个星期一）。\n\n4. 澳大利亚：劳动节是全国性的法定假日，通常放假1天，即劳动节当天（10月的第一个星期一）。\n\n5. 日本：劳动节不是全国性的法定假日，但通常放假1天，即劳动节当天（5月1日）。\n\n\n有些国家需要将劳动节放假期间的工作日进行调休，以保证工作效率。例如中国的劳动节假期前和劳动节假期后会各增加一天假期来进行调整。"
      },
      {
        "query": "各个国家的工人每年的放假时间有多少？",
        "answer": "不同国家的工人每年的放假时间有所不同，以下是一些国家的情况：\n\n1. 中国：法定带薪年假为5天至15天，根据工龄和单位规定有所不同；国家规定的法定节假日有11天。\n\n2. 美国：没有国家规定的带薪年假，通常由雇主和雇员协商决定；法定假日有10天，其中7天是全国性的法定假日。\n\n3. 日本：法定带薪年假为10天至20天，根据工龄和单位规定有所不同；法定节假日有16天。\n\n4. 德国：法定带薪年假为24天至30天，根据工龄和单位规定有所不同；法定节假日有9天。\n\n5. 法国：法定带薪年假为25天，法定节假日有11天。\n\n需要注意的是，虽然有国家规定了法定假日和带薪年假，但在一些职业和行业中，工作需要进行轮班或加班，法定的假日和年假有可能不全是连续的，并不一定能够100%的实现工人享有休假的权利。"
      }
    ]
  }
}
```

#### english

```json
{
  "id": "G8l7KZ65dl",
  "conversation_hash": "f35e40e91ce604ec285a6202d378d818",
  "conversations": {
    "turn": 3,
    "lang": "English",
    "topic": "3",
    "contents": [
      {
        "query": "What is a best practice when approaching an automation effort?\n\n\nprioritizing cost reduction above all other goals\n\n\neliminating the need for a technical support team\n\n\nacquiring new resources before starting automation work\n\n\nimproving the current process before automating it",
        "answer": "improving the current process before automating it."
      },
      {
        "query": "In the context of automation strategy, what is a Center of Excellence?\n\n\na central server where all of an organization’s automation software is hosted\n\n\na team that establishes best practices for automation efforts within an organization\n\n\na team composed of the organization’s most senior Artificial Intelligence programmers\n\n\na community that includes all those within an organization who have created automation programs",
        "answer": "A Center of Excellence (CoE) is a team that establishes best practices for automation efforts within an organization. A CoE provides guidance on the development, deployment, and maintenance of automation solutions, shares knowledge and best practices across the organization, and supports the training and development of automation professionals. The goal of a CoE is to ensure that automation efforts are aligned with the business strategy and that they result in consistent, high-quality solutions."
      },
      {
        "query": "A Project Manager sets up a recurring invite for meetings using a web-based calendar app on a mobile device.\n\nWhat type of automation does this exemplify?\n\n\nHyperautomation\n\n\nIntelligent Automation\n\n\nCognitive Automation\n\n\nPersonal Automation",
        "answer": "Personal Automation. \n\nThe example given involves a person using a web-based calendar app to set up a recurring meeting invite. This is an example of personal automation, which refers to the use of automation tools and techniques by individuals to improve their own productivity and efficiency. The use of a web-based calendar app is a common personal automation practice that can help individuals manage their schedule and streamline the process of scheduling meetings and appointments."
      }
    ]
  }
}
```