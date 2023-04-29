# Chatbot-with-Translation

Chatbot with Translation. Using [databricks/dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b)

Original input text is translated to english for the best performance of Chatbot, and translated back to original language.

# Install

You might need 24GB+ GPU.
If you want to run smaller model, try to find language model in https://huggingface.co/
and change the model in `main.py:16`.

```python
...
generate_model = "{huggingface model}"
...
```

```bash
pip install -r requirements.txt
```

# Usage

```
uvicorn main:app --reload
```

It might take 2~5 min at first.

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
