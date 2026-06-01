# ttp_api/main.py - Match Colab training setup
import os, json, time, uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import uvicorn

HOST = os.getenv('TTP_API_HOST', '127.0.0.1')
PORT = int(os.getenv('TTP_API_PORT', '8002'))
API_KEY = os.getenv('TTP_API_KEY', 'ethiopia-ttp-dev-key')
ADAPTER_PATH = '/Users/hannateshager/Ethiopia_2026election_monitoring_tool/model_cache/gemma-disarm-phase3-ttp'

app = FastAPI(title='Gemma DISARM TTP API')

_model = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is None:
        try:
            from unsloth import FastModel
            from unsloth.chat_templates import get_chat_template
            
            print(f"🔄 Loading from adapter path (matches Colab setup)...")
            
            # EXACT MATCH to Colab notebook loading
            _model, _tokenizer = FastModel.from_pretrained(
                model_name=ADAPTER_PATH,  # Load from YOUR adapter folder
                max_seq_length=4096,      # Match training config
                load_in_4bit=True,        # Match training config
                # Don't specify local_files_only - let Unsloth handle it
            )
            
            # Enable inference mode (from Colab)
            FastModel.for_inference(_model)
            
            # Setup chat template (from Colab)
            _tokenizer = get_chat_template(_tokenizer, chat_template='gemma-4')
            
            print(f"✅ Model loaded successfully from adapter!")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            print("💡 Trying alternative: force_requantize...")
            
            # Fallback: try with force_requantize
            try:
                from unsloth import FastModel
                from unsloth.chat_templates import get_chat_template
                
                _model, _tokenizer = FastModel.from_pretrained(
                    model_name=ADAPTER_PATH,
                    max_seq_length=4096,
                    load_in_4bit=True,
                    force_requantize=True,  # Force match quantization
                )
                
                FastModel.for_inference(_model)
                _tokenizer = get_chat_template(_tokenizer, chat_template='gemma-4')
                print(f"✅ Model loaded with force_requantize!")
                
            except Exception as e2:
                print(f"❌ Also failed: {e2}")
                raise e2
            
    return _model, _tokenizer

class Message(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.1
    max_tokens: int = 1024

def flatten_content(content):
    if isinstance(content, str): return content
    if isinstance(content, list):
        return '\n'.join(item.get('text','') for item in content if isinstance(item, dict) and item.get('type')=='text')
    return str(content)

def build_reply(messages, temperature=0.1, max_tokens=512):
    global _model, _tokenizer
    if _model is None: _model, _tokenizer = get_model()
    
    SYSTEM_PROMPT = "You are a fine-tuned DISARM TTP adjudicator. Consider only these techniques: T0049, T0049.002, T0049.003, T0049.005, T0016, T0060, T0097.202, T0143.003, T0119, T0119.001, T0119.002, T0097.102, T0143.002, T0149.003, T0084.002. Use only raw observable cues present in the dossier. Return strict JSON only. Prefer false negatives over false positives."
    
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    
    normalized = [{'role': m['role'], 'content': flatten_content(m['content'])} for m in full_messages]
    prompt = _tokenizer.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True)
    
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    
    output_ids = _model.generate(
        **inputs, 
        max_new_tokens=max_tokens, 
        temperature=temperature, 
        do_sample=True, 
        use_cache=True
    )
    
    input_len = inputs["input_ids"].shape[1]
    completion = _tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    return completion, int(input_len), int(output_ids.shape[1] - input_len)

@app.post('/v1/chat/completions')
def chat_completions(req: ChatCompletionRequest):
    try:
        print("🤖 Generating TTP analysis with Gemma...")
        text, prompt_tokens, completion_tokens = build_reply(req.messages, req.temperature, req.max_tokens)
        
        return {
            'id': f'chatcmpl-{uuid.uuid4().hex[:12]}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': 'gemma-disarm-phase3-ttp',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': prompt_tokens + completion_tokens}
        }
    except Exception as e:
        print(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health(): return {"status": "ok"}

@app.get('/v1/models')
def list_models(): return {'object': 'list', 'data': [{'id': 'gemma-disarm-phase3-ttp', 'object': 'model'}]}

if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT)
