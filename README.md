# EchoVision: Local AI Chatbot with Image Generation & Multilingual Support

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-darkred.svg)](https://streamlit.io/)
[![Mistral 7B](https://img.shields.io/badge/LLM-Mistral%207B%20Instruct-8000dd?logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjODAwMGRkIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCI+PHJlY3Qgd2lkdGg9IjEwIiBoZWlnaHQ9IjEwIiByeD0iMyIgLz48dGV4dCB4PSI1IiB5PSI3IiBmb250LXNpemU9IjciIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IndoaXRlIj5NTDwvdGV4dD48L3N2Zz4=)](https://mistral.ai/news/announcing-mistral-7b/)
[![Stable Diffusion v1.5](https://img.shields.io/badge/diffusion-Stable%20Diffusion%20v1.5-16B9FA?logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjMTZCOUZBIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCI+PHJlY3Qgd2lkdGg9IjEwIiBoZWlnaHQ9IjEwIiByeD0iMyIvPjx0ZXh0IHg9IjUiIHk9IjciIGZvbnQtc2l6ZT0iNyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0id2hpdGUiPlNEPC90ZXh0Pjwvc3ZnPg==)](https://huggingface.co/runwayml/stable-diffusion-v1-5)
[![Status: Active](https://img.shields.io/badge/status-active-brightgreen.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-blue.svg?style=flat-square)](https://makeapullrequest.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Welcome to EchoVision, an advanced local AI chatbot combining the power of the Mistral 7B large language model with Stable Diffusion for on-device conversational AI and image generation. Designed for efficiency and usability on consumer hardware (like RTX 3050 GPUs), EchoVision supports multiple languages and features a sleek web interface.

---

## Features
- Conversational AI powered by Mistral 7B Instruct (local GGUF model)
- High-quality image generation with Stable Diffusion v1.5 (local safetensors)
- Multilingual support: English, Hindi, Spanish, and French with automatic language detection
- Conversation memory with token-based context management
- Rich, interactive UI built with Streamlit
- Optimized for consumer GPUs (e.g., RTX 3050 with 4GB VRAM)
- Session management with persistent conversation histories
- Configurable prompt system with strict image generation triggers
- Robust error handling and informative logging
- CORS-enabled FastAPI backend for flexible deployment

---

## Project Structure
```
EchoVision/
│
├── main.py                      # FastAPI backend server (root)
├── config.py                    # Project-wide configuration and environment
├── requirements.txt             # Python dependencies
│
├── core/
│   ├── llm/
│   │   ├── mistral_handler.py   # Mistral LLM integration and inference
│   │   └── memory.py            # Conversation memory and token management
│   ├── image_generation/
│   │   └── stable_diffusion.py  # Stable Diffusion image generation handler
│   └── utils/
│       └── language_handler.py # Multi-language detection and prompt creation
│
├── frontend/
│   └── streamlit_app.py         # Interactive chat frontend with image display
│
├── data/                        # Stores conversation session files & generated images
├── models/                      # Local model files - Mistral GGUF and SD safetensors
│
└── .env                        # Optional environment configuration file (paths, etc.)
```

---

## Prerequisites
- Python 3.10 environment
- NVIDIA GPU (RTX 3050 recommended) with CUDA 11.8 or compatible
- At least 16GB RAM
- Models downloaded locally:
  - Mistral 7B GGUF: ../../../local_models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf
  - Stable Diffusion v1.5 safetensors: ../../../local_models/stable-diffusion/v1-5-pruned-emaonly.safetensors

---

## Installation & Setup
1. Clone this repo
```
git clone https://github.com/Hartz-byte/EchoVision.git
cd EchoVision
```

2. Create and activate Python 3.10 venv
```
python -m venv venv
source venv/bin/activate              # Linux/macOS
venv\Scripts\activate.bat             # Windows CMD
source venv/Scripts/activate          # Windows PowerShell/Git Bash
```

3. Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

4. Download and place the LLM and SD models
Place the models in the specified local paths (can be updated in config.py or .env).

5. Run the backend server
```
python main.py
```

6. Run the frontend UI
In another terminal (with venv activated):
```
streamlit run frontend/streamlit_app.py
```
Backend: http://localhost:8000

Frontend: http://localhost:8501

---

## Configuration
Modify config.py or .env for:
- Model paths
- GPU and threading parameters
- Memory token limits
- Image generation parameters (resolution, guidance scale, steps)
- Supported languages
- API host and port

---

## Usage
- Type your queries or commands in English, Hindi, Spanish, or French.
- To request images, use explicit phrases like "Create an image of..." or "Generate a picture of..."
- The chatbot responds intelligently; images are generated only when explicitly requested.
- Conversation memory manages context across your session.
- Use the sidebar to monitor API status, clear chat, start new sessions, and see stats.

---

## Technical Details
- Backend: FastAPI with asynchronous endpoints serving chat and image generation.
- LLM Integration: Uses llama-cpp-python to run Mistral 7B GGUF model efficiently with GPU offloading.
- Image Generation: Uses HuggingFace diffusers with VRAM optimization features (sequential CPU offloading, attention slicing).
- Language Detection: Implements robust multi-language detection with langdetect and regex cleaning.
- Memory Management: Token-aware conversation memory with trimming to avoid token limit overflow.
- Frontend: Streamlit app with custom CSS for message bubbles, aligned chat messages, image previews capped at 300px, and real-time API status.

---

## Troubleshooting
- Ensure CUDA drivers and PyTorch with CUDA support are correctly installed.
- Verify local model files are correctly named and paths are accurate.
- Use the sidebar "Clear Chat" and "New Session" buttons to reset conversation states.
- Check API logs for errors during model loading or generation.
- Increase the timeout in the frontend if image generation takes longer on your hardware.

---

## Credits
- Mistral AI (Mistral 7B model)
- HuggingFace (diffusers library & Stable Diffusion)
- Streamlit community for UI inspiration
- Open source developers of PyTorch, FastAPI, and associated Python tooling

---

## ⭐️ Give it a Star
If you found this repo helpful or interesting, please consider giving it a ⭐️. It motivates me to keep learning and sharing!

---
