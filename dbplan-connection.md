# DB Plan: Connect MySQL / phpMyAdmin to Production ML App and Containerize

## Purpose
This plan describes how to connect your live MySQL database, managed through phpMyAdmin, to your production Python ML application. It explains the flow from fetching sensor records, converting them into model input, and packaging the app in Docker.

## 1. Production Architecture

Components:
- `MySQL` database holds sensor readings and application state
- `phpMyAdmin` is used only for database administration and verification
- Python backend reads rows from MySQL and feeds them into the ML model
- Docker packages the backend and its dependencies for production deployment

## 2. Use Existing Production Code

### Backend source files in this repo
- `api/app.py` — Flask API and frontend routing
- `model2/src/predict.py` — model2 prediction logic and feature construction
- `model3/predict_crop.py` — crop prediction logic and model artifact loading

The integration point is in `api/app.py`: the API receives requests, reads database records, and forwards data to the prediction module.

## 3. MySQL Connection Strategy

Use SQLAlchemy or `mysql-connector-python` in the production Python service.
Store credentials in environment variables, never in source code.

Required environment variables:
- `MYSQL_HOST`
- `MYSQL_PORT`
- `MYSQL_USER`
- `MYSQL_PASSWORD`
- `MYSQL_DATABASE`

Example connection URL:
```python
from sqlalchemy import create_engine

mysql_url = (
    f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
)
engine = create_engine(mysql_url)
```

## 4. Fetch Live Sensor Data from phpMyAdmin-managed MySQL

phpMyAdmin does not change how the application connects. It only provides a browser UI to inspect and manage the same MySQL database.

In production, read the live table directly from the database using the same connection string your phpMyAdmin session uses.

### Real-time data retrieval flow
1. `api/app.py` receives a request for prediction or a scheduled ingestion event.
2. The backend opens a MySQL connection using environment configuration.
3. It executes a query to retrieve the latest sensor rows or a specific device record.
4. It converts that row(s) to the exact feature set expected by the model.
5. It passes the features into `model2/src/predict.py` or `model3/predict_crop.py`.
6. It returns the prediction response.

## 5. Integrating Database Reads with Production ML Code

Use the live row data as direct input to the production prediction function.

### Example integration pattern
```python
from sqlalchemy import create_engine
import pandas as pd
from model3.predict_crop import predict_crop

engine = create_engine(mysql_url)
query = "SELECT nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall FROM sensor_readings WHERE device_id = :device_id ORDER BY timestamp DESC LIMIT 1"

sensor_df = pd.read_sql(query, engine, params={"device_id": device_id})
row = sensor_df.iloc[0]

crop, confidence = predict_crop(
    nitrogen=row['nitrogen'],
    phosphorus=row['phosphorus'],
    potassium=row['potassium'],
    temperature=row['temperature'],
    humidity=row['humidity'],
    ph=row['ph'],
    rainfall=row['rainfall'],
)
```

This is the production pattern: the backend reads actual database fields and maps them to the model function.

## 6. Use Environment Variables in Production

Set the database and app settings outside of code.

Example variables for Docker:
```text
MYSQL_HOST=mysql
MYSQL_PORT=3306
MYSQL_USER=app_user
MYSQL_PASSWORD=app_password
MYSQL_DATABASE=sensor_ml
FLASK_ENV=production
```

## 7. Dockerize the Production App

Create a `Dockerfile` that installs dependencies and runs the Flask API.

### Dockerfile
```Dockerfile
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

ENV MYSQL_HOST=mysql
ENV MYSQL_PORT=3306
ENV MYSQL_USER=app_user
ENV MYSQL_PASSWORD=app_password
ENV MYSQL_DATABASE=sensor_ml

EXPOSE 5000
CMD ["python", "api/app.py"]
```

## 8. Docker Compose for MySQL + App

If you want a production-like local stack, use Docker Compose.

### docker-compose.yml
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5000:5000"
    depends_on:
      - mysql
    environment:
      MYSQL_HOST: mysql
      MYSQL_PORT: 3306
      MYSQL_USER: app_user
      MYSQL_PASSWORD: app_password
      MYSQL_DATABASE: sensor_ml

  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: sensor_ml
      MYSQL_USER: app_user
      MYSQL_PASSWORD: app_password
    volumes:
      - mysql_data:/var/lib/mysql

volumes:
  mysql_data:
```

## 9. Deployment Notes

- `phpMyAdmin` is only for database management; the app connects directly to MySQL.
- The production service should query live rows, not static sample data.
- Use the same table and column names that exist in your database schema.
- If your model expects a specific feature order, ensure the query returns those fields consistently.

## 10. Final Plan Summary

1. Keep `phpMyAdmin` for administration only.
2. Configure MySQL connection in the app using environment variables.
3. Fetch live rows from the production table using SQL queries.
4. Map database columns to the ML prediction function inputs.
5. Run the Flask application inside Docker.
6. Use `docker build` and `docker run` for production deployment.

This file is the production plan for connecting your MySQL data source directly to the ML model and containerizing the result.