FROM python:3.11-slim
WORKDIR /app
COPY requirments.txt .
RUN pip install --no-cache-dir -r requirments.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
LABEL authors="HP"