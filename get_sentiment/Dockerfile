FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ./
RUN python3.9 -m pip install -r requirements.txt -t .

COPY get_sentiment.py ./
COPY cnn_model.pt /opt/ml/cnn_model.pt

CMD ["get_sentiment.lambda_handler"]
