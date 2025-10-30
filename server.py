from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
chat_pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
summary_pipe = pipeline("summarization", model="Falconsai/text_summarization")

# Thread pool for running model inference in background
executor = ThreadPoolExecutor(max_workers=4)

class ChatRequest(BaseModel):
    text: str

class SummaryRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Inference</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            .toggle { display: flex; gap: 10px; margin: 20px 0; }
            .toggle button { padding: 10px 20px; background: #ddd; border: none; cursor: pointer; }
            .toggle button.active { background: #007bff; color: white; }
            textarea { width: 100%; padding: 10px; margin: 10px 0; box-sizing: border-box; }
            .submit { padding: 10px 20px; background: #28a745; color: white; border: none; cursor: pointer; }
            .submit:hover { background: #218838; }
            .output { margin-top: 20px; padding: 15px; background: #f5f5f5; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h1>LLM Inference App</h1>

        <div class="toggle">
            <button id="chatBtn" class="active" onclick="switchMode('chat')">Chat Generation</button>
            <button id="summaryBtn" onclick="switchMode('summary')">Text Summarization</button>
        </div>

        <textarea id="input" rows="6" placeholder="Enter your text..."></textarea>
        <button class="submit" onclick="submit()">Submit</button>
        <div id="output" class="output" style="display:none;"></div>

        <script>
            let mode = 'chat';

            function switchMode(newMode) {
                mode = newMode;
                document.getElementById('chatBtn').classList.toggle('active', mode === 'chat');
                document.getElementById('summaryBtn').classList.toggle('active', mode === 'summary');
                document.getElementById('input').placeholder = mode === 'chat' ?
                    'Enter your question...' : 'Enter text to summarize...';
            }

            async function submit() {
                const text = document.getElementById('input').value;
                const output = document.getElementById('output');
                output.textContent = mode === 'chat' ? 'Generating...' : 'Summarizing...';
                output.style.display = 'block';

                const endpoint = mode === 'chat' ? '/chat' : '/summarize';
                const res = await fetch(endpoint, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text})
                });
                const data = await res.json();
                output.textContent = mode === 'chat' ? data.response : data.summary;
            }
        </script>
    </body>
    </html>
    """

def run_chat(text: str):
    out = chat_pipe(text, max_new_tokens=100, return_full_text=False)
    return out[0]["generated_text"]

def run_summarize(text: str):
    out = summary_pipe(text, max_length=130, min_length=30, do_sample=False)
    return out[0]["summary_text"]

@app.post("/chat")
async def chat(req: ChatRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, run_chat, req.text)
    return {"response": result}

@app.post("/summarize")
async def summarize(req: SummaryRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, run_summarize, req.text)
    return {"summary": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
