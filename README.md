# Interqu-audio - API

## Functionalities
- Text2Speech - Uses google API (no api key) to convert speech to text as opposed to the pytorch branch which uses huggingface's Text2Speech model.

## Audio Sample Sources
- [16khz](http://www.fit.vutbr.cz/~motlicek/speech_hnm.html) - audio1, audio2, audio3
- [16khz Sound Demo for the Wu-Wang](https://web.cse.ohio-state.edu/~wang.77/pnl/demo/WuReverb.html)

## Resources
- [aws s3 get file using lambda function](https://stackoverflow.com/questions/30651502/how-to-get-contents-of-a-text-file-from-aws-s3-using-a-lambda-function)

## Serverless Lambda

Install libraries with precompiled binaries
```python
pip install \
    --platform manylinux2014_aarch64 \
    --target=your-lambda-function\
    --implementation cp \
    --python 3.9 \
    --only-binary=:all: --upgrade \
    -r requirements.txt
```

Updating your function from AWS cli
```bash
aws lambda update-function-code --function-name MyLambdaFunction --zip-file fileb://my-deployment-package.zip
```
