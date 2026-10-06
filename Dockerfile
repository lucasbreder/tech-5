FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Compilador de contingência para dependências científicas sem wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Artefatos reproduzíveis: a imagem não depende de joblibs/chroma do host.
RUN python main.py train && python main.py rag-ingest

EXPOSE 8501

# Healthcheck simples da API Streamlit
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD \
    python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).status==200 else 1)" || exit 1

# Jornada completa pela interface Streamlit
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
