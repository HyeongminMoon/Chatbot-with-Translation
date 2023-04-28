from transformers import pipeline
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import translators as ts
import json
import os
from datetime import datetime
import string
import random
from langdetect import detect

app = FastAPI()
result_save_dir = "test_results"
hash_candidates = string.ascii_letters + string.digits
generate_model = "databricks/dolly-v2-12b"

class DefaultTS():
    ts_tool = "papago"
    back_ts_tool = "google"


class PromptData(BaseModel):
    text: str


def check_lang(text: str):
    return detect(text)


def save_result(result: dict):
    now = datetime.now()
    date_string = now.strftime("%Y%m%d_%H%M%S")
    hashes = "".join(random.choice(hash_candidates) for _ in range(4))

    with open(os.path.join(result_save_dir, date_string + hashes + ".json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


@app.on_event("startup")
def startup_event():
    global instruct_pipeline
    global ts_pool
    global STARTUP
    STARTUP = False
    instruct_pipeline = pipeline(
        model=generate_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    ts.preaccelerate()
    ts_pool = ts.translators_pool
    STARTUP = True


@app.post("/predict/{ts_tool}")
def predict(data: PromptData, ts_tool: str):
    if not STARTUP:
        return {"error": "Sever is starting. It takes ~2min for optimization."}
    if ts_tool not in ts_pool:
        return {"error": f"There is no matched name of translator. available:{ts_pool}"}
    try:
        ori_lang = check_lang(data.text)
        if ori_lang == 'en':
            result = instruct_pipeline(data)[0]
            # result["generated_text"]
        else:
            q_text = data.text
            ts_text = ts.translate_text(
                q_text, translator=DefaultTS.ts_tool, if_use_preacceleration=True,
                to_language='en', timeout=10
            )
            result = instruct_pipeline({"text": ts_text})[0]
            gen_text = result["generated_text"]
            ts_gen_text = ts.translate_text(
                gen_text, translator=ts_tool, if_use_preacceleration=True,
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
    except Exception as e:
        return {"error": repr(e)}


@app.post("/predict")
def predict(data: PromptData):
    ts_tool = DefaultTS.ts_tool
    back_ts_tool = DefaultTS.back_ts_tool
    if not STARTUP:
        return {"error": "Sever is starting. It takes ~2min for optimization."}
    try:
        ori_lang = check_lang(data.text)
        if ori_lang == 'en':
            result = instruct_pipeline(data)[0]
            # result["generated_text"]
        else:
            q_text = data.text
            ts_text = ts.translate_text(
                q_text, translator=DefaultTS.ts_tool, if_use_preacceleration=True,
                to_language='en', timeout=10
            )
            result = instruct_pipeline({"text": ts_text})[0]
            gen_text = result["generated_text"]
            ts_gen_text = ts.translate_text(
                gen_text, translator=ts_tool, if_use_preacceleration=True,
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
    except Exception as e:
        return {"error": repr(e)}


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
                q_text, translator=ts_tool, if_use_preacceleration=True
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