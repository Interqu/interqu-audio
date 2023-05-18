# Execution role arn:aws:iam::537801392411:role/InterquLambdaS3READ

import json, boto3
import speech_recognition as sr
import os

s3 = boto3.client("s3")


def handler(event, context):
    # body = json.loads(event["body"])
    body = event["body"]
    bucket = body["bucket"]
    file_name = body["file_name"]

    # retrieving the file
    if bucket:
        s3.download_file(bucket, file_name, "/tmp/audio.wav")
<<<<<<< Updated upstream:serverless/speech2text/interqu-speech2text.py
        file_size = os.path.getsize("/tmp/audio.wav")
        return file_size

=======
>>>>>>> Stashed changes:serverless/text2speech/interqu-speech2text.py
    else:
        return f"No bucket name provided"

    r = sr.Recognizer()
    with sr.AudioFile("/tmp/audio.wav") as source:
        audio = r.record(source)
        s = r.recognize_google(audio)

    return {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(s, indent=4),
    }
