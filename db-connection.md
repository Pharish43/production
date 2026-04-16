# MySQL Database Connection for Sensor Data and Containerized ML Production

## Overview
This guide explains how to connect a MySQL database to your machine learning model so sensor data can be loaded as model input, then how to containerize the production pipeline.

The recommended architecture:
- MySQL stores sensor readings and metadata.
- Python service reads sensor rows from MySQL, preprocesses them, and passes them to the ML model.
- The service is packaged in Docker for production.

## 1. MySQL Database Setup

1. Install MySQL on your machine or use a hosted MySQL service.
2. Create a database for sensor data. Example:
```sql
CREATE DATABASE sensor_ml;
USE sensor_ml;
```
3. Create a table for incoming sensor records. Example:
```sql
CREATE TABLE sensor_readings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    device_id VARCHAR(64) NOT NULL,
    timestamp DATETIME NOT NULL,
    temperature FLOAT,
    humidity FLOAT,
    pressure FLOAT,
    nitrogen FLOAT,
    phosphorus FLOAT,
    potassium FLOAT,
    rainfall FLOAT,
    ph FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
4. Insert sample rows or stream data from your sensors into this table.

## 2. Install Python MySQL Connector

Add a Python connector to `requirements.txt`:
```text
mysql-connector-python
pandas
SQLAlchemy
```

Or install directly:
```bash
pip install mysql-connector-python pandas SQLAlchemy
```

## 3. Connect Python to MySQL

Use SQLAlchemy for a clean connection and easy query handling.

### Example connection code
```python
from sqlalchemy import create_engine
import pandas as pd

DB_USER = 'your_user'
DB_PASS = 'your_password'
DB_HOST = '127.0.0.1'
DB_PORT = 3306
DB_NAME = 'sensor_ml'

connection_url = (
    f'mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
)
engine = create_engine(connection_url)

query = '''
SELECT device_id, timestamp, temperature, humidity, pressure,
       nitrogen, phosphorus, potassium, rainfall, ph
FROM sensor_readings
WHERE timestamp >= NOW() - INTERVAL 1 HOUR
ORDER BY timestamp DESC
LIMIT 100;
'''

sensor_df = pd.read_sql(query, engine)
```

## 4. Preprocess Sensor Data for the Model

Transform database rows into the input shape your ML model expects.

### Example preprocessing
```python
def build_model_input(sensor_df):
    sensor_df = sensor_df.dropna(subset=['temperature', 'humidity', 'nitrogen', 'phosphorus', 'potassium', 'ph', 'rainfall'])
    # Example feature selection
    features = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
    model_input = sensor_df[features]
    return model_input

X = build_model_input(sensor_df)
```

If your model expects a single row, choose the latest reading or aggregate sensor values before prediction.

## 5. Feed the Data to the ML Model

Use your existing model prediction function.

### Example prediction flow
```python
from model2.src.predict import predict_crop

input_row = X.iloc[0].to_dict()

crop, confidence = predict_crop(
    nitrogen=float(input_row['nitrogen']),
    phosphorus=float(input_row['phosphorus']),
    potassium=float(input_row['potassium']),
    temperature=float(input_row['temperature']),
    humidity=float(input_row['humidity']),
    ph=float(input_row['ph']),
    rainfall=float(input_row['rainfall']),
)

print('Recommended crop:', crop)
print('Confidence:', confidence)
```

## 6. Containerize the App

Create a `Dockerfile` for your ML application.

### Example `Dockerfile`
```Dockerfile
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

ENV MYSQL_HOST=mysql
ENV MYSQL_PORT=3306
ENV MYSQL_USER=your_user
ENV MYSQL_PASSWORD=your_password
ENV MYSQL_DATABASE=sensor_ml

CMD ["python", "api/app.py"]
```

### Example `docker-compose.yml`
```yaml
version: '3.8'
services:
  app:
    build: .
    depends_on:
      - mysql
    environment:
      - MYSQL_HOST=mysql
      - MYSQL_PORT=3306
      - MYSQL_USER=your_user
      - MYSQL_PASSWORD=your_password
      - MYSQL_DATABASE=sensor_ml
    ports:
      - "5000:5000"

  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD:rootpassword
      MYSQL_DATABASE: sensor_ml
      MYSQL_USER: your_user
      MYSQL_PASSWORD: your_password
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql

volumes:
  mysql_data:
```

## 7. Build and Run the Container

### Build locally
```bash
docker build -t <your-user>/<repo-name>:latest .
```

### Run with Docker Compose
```bash
docker-compose up --build
```

### Run standalone
```bash
docker run --rm -p 5000:5000 \
  -e MYSQL_HOST=<host> \
  -e MYSQL_PORT=3306 \
  -e MYSQL_USER=<user> \
  -e MYSQL_PASSWORD=<password> \
  -e MYSQL_DATABASE=sensor_ml \
  <your-user>/<repo-name>:latest
```

## 8. Production Deployment Plan

1. Use environment variables for all DB credentials.
2. Never hardcode passwords inside source files or the image.
3. Use secrets management or Docker secrets for production.
4. If the model requires scheduled data, create a polling or event-based ingestion layer.
5. Add logging and error handling for database failures.
6. Monitor query latency and model latency.
7. Keep the MySQL service separate from the ML service, even in containerized deployments.

## 9. Clear Plan Summary

1. Create MySQL database and sensor table.
2. Add Python DB connector dependencies.
3. Query sensor rows using SQLAlchemy.
4. Preprocess the data into model input format.
5. Call the ML model prediction function.
6. Create Dockerfile and optional docker-compose.yml.
7. Build the image and run the container.
8. Use environment variables for production credentials.

## 10. Helpful Tips
- If your model uses batch data, write a small loader that fetches multiple rows.
- Use `pandas.read_sql()` for quick prototyping and batch reads.
- If you need low latency, limit queries to the latest sensor row.
- For multiple devices, include `device_id` or `timestamp` filtering.
