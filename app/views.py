from app import app
import os

import json
from flask import Flask, request, jsonify
import logging

from app.demographics import Demographics

from app.utils import download_artifacts
from app.config import PORT


download_artifacts()
 
demographic_predictor = Demographics()

@app.route("/")
def index():

    # Use os.getenv("key") to get environment variables
    app_name = os.getenv("APP_NAME")

    if app_name:
        return f"Hello from {app_name} running in a Docker container behind Nginx!"

    return "Hello from Flask"



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    content = request.json
    result = demographic_predictor.get_demographics(content)
    return jsonify(result)