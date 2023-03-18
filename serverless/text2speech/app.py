import speech_recognition as sr

def handler(event, context):
    body = json.loads(event["body"])
    seed = body["seed"]
    
    
    
    # https://docs.aws.amazon.com/apigateway/latest/developerguide/lambda-proxy-binary-media.html
    return {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": {"Content-Type": "image/png"},
        "body": base64.b64encode(im_file.getvalue()),
    }

ls
ls