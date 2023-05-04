# Chatbot-with-Translation

Chatbot with Translation. Using [databricks/dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b)

Original input text is translated to english for the best performance of Chatbot, and translated back to original language.

# Install

You might need 24GB+ GPU.
If you want to run smaller model, try to find language model in https://huggingface.co/
and change the model in `single.py`.

```python
...
generate_model = "{huggingface model}"
...
```

```bash
pip install -r requirements.txt
```

The smaller model for chatbot-mode is a little more demanding. You will need to change the model in `chatbot.py` and also prepare suitable langchain pipeline.

# Usage

### Single Question & Answering

```bash
uvicorn single:app --reload
```

It might take 2~5 min at first.

### Chatbot mode

You need langchain to run chatbot mode.

```bash
pip install langchain
uvicorn chatbot:app --reload
```

# Example

```python
import requests
ts_tool = 'google'
url = f"http://localhost:8000/predict"
data = {"text": "30명이 함께 즐길 수 있는 스포츠를 10가지 알려줘"}

response = requests.post(url, json=data)
result = response.json()
print('result:\n', result['generated_text'])
```

```bash
result:
 다음은 30명이 함께 즐길 수 있는 스포츠입니다:
1. 축구
2. 축구
3. 하키
4. 크리켓
5. 배구
6. 골프
7. 테니스
8. 탁구
9. 야구
10. 탁구
```

### Japanese Translated Example

```bash
Human:  プロセスマイニングとは？
result:
 データをマイニングするときは、関心のあるアクション（たとえば、質問に答える、パターンを発見する、または構造を特定する）を指定します。ソフトウェアは、データの収集と分析のプロセスを自動化して、申し立ての証拠を見つけます。


translated question:
 What is process mining?
original result:
When you mine data, you specify an action of interest (e.g., answer a question, discover a pattern, or identify a structure) and the software automates the process of collecting and analyzing data to find a proof of the allegation.
```

### Urdu Translated Example

```bash
Human:  ٹیلی فون کس نے ایجاد کیا؟
result:
 ٹیلیفون کے تصور کی ایجاد 1876 میں الیگزینڈر گراہم بیل نے کی تھی۔ پہلا ورکنگ ٹیلیفون 1877 میں تھامس ایڈیسن نے تعمیر کیا تھا۔ ٹیلیفون نے صوتی پیغامات بھیجنے اور وصول کرنے کا تقریبا مفت طریقہ فراہم کرکے مواصلات میں انقلاب برپا کردیا اور 20 ویں صدی میں گھروں اور کاروباروں میں جلدی سے ہر جگہ عام ہوگیا۔ ٹیلیفون کی صلاحیتوں میں موبائل اور فکسڈ براڈ بینڈ نیٹ ورکس پر ٹیکسٹ میسجنگ ، ویڈیو کانفرنسنگ ، انٹرنیٹ تک رسائی ، اور دیگر صلاحیتوں کو شامل کرنے میں اضافہ ہوا ہے۔


translated question:
 Who invented the telephone?
original result:
Invention of the telephone concept was done by Alexander Graham Bell in 1876.  The first working telephone was built in 1877 by Thomas Edison.  The telephone revolutionized communications by providing a nearly free way to send and receive voice messages and quickly became ubiquitous in homes and businesses throughout the 20th century. The telephones capabilities have grown to include text messaging, video conferencing, Internet access, and other capabilities over mobile and fixed broadband networks.
```
