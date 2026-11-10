FROM python:3.9-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements-app.txt .
RUN pip install --no-cache-dir --user -r requirements-app.txt

COPY --chown=user config.yaml ./
COPY --chown=user src ./src
COPY --chown=user app ./app
COPY --chown=user models/live_model.joblib models/live_model_info.json ./models/
ENV PORT=10000
EXPOSE 10000
CMD uvicorn app.server:app --host 0.0.0.0 --port $PORT
