#Dockerfile to build a flask app

FROM python:3.8-slim-buster
ADD . /api
WORKDIR /api
RUN pip install -r requirements.txt


CMD ["python", "api.py"]
