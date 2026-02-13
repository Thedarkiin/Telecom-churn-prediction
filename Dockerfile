FROM python:3.12

WORKDIR /app

# Upgrade pip to the latest version immediately
RUN pip install --upgrade pip

# Copy and install (letting pip resolve the best versions)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "run_app.py"]