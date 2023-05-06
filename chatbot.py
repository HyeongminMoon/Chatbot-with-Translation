import torch
from fastapi import FastAPI, Request
import translators as ts
from translators.server import TranslatorError
import json
import os
from datetime import datetime
import string
import random
from langdetect import detect

# for history mode
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from langchain import LLMChain
from langchain.llms import HuggingFacePipeline

from pipelines.dolly.generate import InstructionTextGenerationPipeline
from pipelines.base.defines import DefaultTS, PromptData
from pipelines.base.memory import ChatbotMemory
from pipelines.base.prompt_template import chatbot_prompt

app = FastAPI()
result_save_dir = "test_results"
hash_candidates = string.ascii_letters + string.digits
generate_model = "databricks/dolly-v2-12b"


def check_lang(text: str):
    return detect(text)


def save_result(result: dict):
    now = datetime.now()
    date_string = now.strftime("%Y%m%d_%H%M%S")
    hashes = "".join(random.choice(hash_candidates) for _ in range(4))

    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)

    with open(os.path.join(result_save_dir, date_string + hashes + ".json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


@app.on_event("startup")
def startup_event():
    global chat_chain
    global ts_pool
    global STARTUP
    STARTUP = False
    tokenizer = AutoTokenizer.from_pretrained(generate_model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        generate_model, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    pipe = InstructionTextGenerationPipeline(
        model=model, tokenizer=tokenizer, return_full_text=True,
        task='text-generation'
    )
    hf_pipeline = HuggingFacePipeline(pipeline=pipe)
    chat_chain = LLMChain(
        llm=hf_pipeline,
        prompt=chatbot_prompt,
    )
    # ts.preaccelerate()
    ts_pool = ts.translators_pool
    STARTUP = True


@app.post("/predict/{ts_tool}")
def predict(data: PromptData, request: Request, ts_tool: str, back_ts_tool: str = None):
    if not STARTUP:
        return {"error": "Sever is starting. It takes ~2min for optimization."}
    if ts_tool not in ts_pool:
        return {"error": f"There is no matched name of translator. available:{ts_pool}"}
    if back_ts_tool is None:
        back_ts_tool = ts_tool
    if data.text == "clear":
        user_id = request.client.host
        ChatbotMemory.clear_memory(user_id)
        return {"info": "the model's memory is cleared."}
    try:
        user_id = request.client.host
        ori_lang = check_lang(data.text)
        result = {}
        if ori_lang == 'en':
            history = ChatbotMemory.load_context(user_id)['history']
            output = chat_chain.predict(human_input=data.text, history=history)
            ChatbotMemory.save_context(user_id, data.text, output)
            result["generated_text"] = output
        else:
            q_text = data.text
            ts_text = ts.translate_text(
                q_text, translator=DefaultTS.ts_tool, if_use_preacceleration=False,
                to_language='en', timeout=10
            )
            history = ChatbotMemory.load_context(user_id)['history']
            gen_text = chat_chain.predict(human_input=ts_text, history=history)
            ChatbotMemory.save_context(user_id, ts_text, gen_text)
            ts_gen_text = ts.translate_text(
                gen_text, translator=ts_tool, if_use_preacceleration=False,
                from_language='en', to_language=ori_lang, timeout=10
            )
            result["original_generated_text"] = gen_text
            result["generated_text"] = ts_gen_text
            result["translator"] = ts_tool
            result["back_translator"] = ts_tool
            result["translated_text"] = ts_text
            result["detected_lang"] = ori_lang

        result["detected_lang"] = ori_lang
        result["generate_model"] = generate_model  
        result["original_text"] = data.text
        save_result(result)
        return result
    # except LangDetectException:
    except TranslatorError:
        user_id = request.client.host
        ori_lang = check_lang(data.text)
        result = {"info": "No features in text. Model uses original prompt automatically."}
        history = ChatbotMemory.load_context(user_id)['history']
        output = chat_chain.predict(human_input=data.text, history=history)
        ChatbotMemory.save_context(user_id, data.text, output)
        result["generated_text"] = output
        result["generate_model"] = generate_model  
        result["original_text"] = data.text
        save_result(result)
        return result

    except Exception as e:
        return {"error": repr(e)}


@app.post("/predict")
def _predict(data: PromptData, request: Request):
    return predict(data, request, DefaultTS.ts_tool, DefaultTS.back_ts_tool)


@app.post("/translate/{ts_tool}")
def translate(data: PromptData, ts_tool: str):
    ts_tool = DefaultTS.ts_tool
    if not STARTUP:
        return {"error": "Sever is starting. It takes ~2min for optimization."}
    try:
        ori_lang = check_lang(data.text)
        if ori_lang == 'en':
            pass
        else:
            result = {}
            q_text = data.text
            translated_text = ts.translate_text(
                q_text, translator=ts_tool, if_use_preacceleration=False
            )
            result["original_text"] = data.text
            result["translator"] = ts_tool
            result["translated_text"] = translated_text
            save_result(result)

        result["detected_lang"] = 'en'
        result["original_text"] = data.text
        return result
    except Exception as e:
        return {"error": repr(e)}