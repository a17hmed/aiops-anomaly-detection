# AIops-Based Cloud Resource Anomaly Detection System

## Overview
This project implements an AIops system to detect anomalies in cloud resource usage using machine learning. The solution is deployed using FastAPI, Docker, and Kubernetes with health checks and autoscaling.

## Features
- Supervised ML anomaly detection
- REST API using FastAPI
- Docker containerization
- Kubernetes deployment
- Liveness & Readiness probes
- Horizontal Pod Autoscaler (HPA)

## Tech Stack
- Python, Scikit-learn
- FastAPI
- Docker
- Kubernetes

## Run Locally
```bash
docker build -t aiops-anomaly-api .
docker run -p 8000:8000 aiops-anomaly-api
