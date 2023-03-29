# Execution role arn:aws:iam::537801392411:role/InterquLambdaS3READ

import json, boto3
import speech_recognition as sr

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
        try:
            with open ('/tmp/audio.wav', 'wb') as f:
                s3.download_fileobj(bucket, file_name, f)
        except Exception as e:
            return f"Exception: {e}"
    else:
        return -1

    r = sr.Recognizer()
    with sr.AudioFile("/tmp/audio.wav") as source:
        audio = r.record(source)
    try:
        s = r.recognize_google(audio)
    except Exception as e:
        return f"Exception during translation: {e}"

    return {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(s, indent=4),
    }
