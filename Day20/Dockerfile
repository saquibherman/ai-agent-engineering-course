# Image for the Day20_P6 agent service (FastAPI + LangChain agent + MCP + RAG).
# Build context is Day20/ (see docker-compose.yml), so paths start at Day20_P6/.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes
COPY Day20_P6/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code, MCP server (server.py) and RAG source (notes.txt); .env is excluded by .dockerignore
COPY Day20_P6/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
