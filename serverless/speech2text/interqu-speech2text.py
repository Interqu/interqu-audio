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

    # see if i can get bucket names
    # buckets = s3.list_buckets()

    # for bucket in buckets["Buckets"]:
    #     bucket["CreationDate"] = bucket["CreationDate"].strftime("%Y-%m-%d %H:%M:%S")
        
    # return {"statusCode": 200, "body": buckets}

    # retrieving the file
    if bucket:
        s3.download_file(bucket, file_name, "/tmp/audio.wav")
        file_size = os.path.getsize("/tmp/audio.wav")
        return file_size

    else:
        return -1

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
