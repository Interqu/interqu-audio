# Execution role arn:aws:iam::537801392411:role/InterquLambdaS3READ

import json, boto3
import speech_recognition as sr
s3 = boto3.client('s3')

def handler(event, context):
    # body = json.loads(event["body"])
    body = event["body"]
    
    return event

    bucket = body["bucket"]
    key = body["key"]
    link = body["link"]
    
    # retrieving the file
    if (key and bucket):
        # prefer retrieving from s3
        response = s3.get_object(Bucket=bucket, Key=key)
        with open("audio.wav", 'wb') as f:
            f.write(contents)
    else:
        # do nothing for now
        print("no key, no body")
        return -1;
    
    r = sr.Recognizer()
    with sr.AudioFile("audio.wav") as source:
        audio = r.record(source)
    try:
        s = r.recognize_google(audio)
        print("Text: "+s)
    except Exception as e:
        print("Exception: "+str(e))
        
    return {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(s, indent=4),
    }