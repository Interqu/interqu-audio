import json
import numpy as np
import PIL.Image
import torch
import pickle
import base64
from io import BytesIO


def handler(event, context):
    print(f"event: {event}")
    print(f"context: {context}")

    body = json.loads(event["body"])
    seed = body["seed"]

    # setting device
    device = torch.device("cpu")

    # loading the model
    with open("stylegan3-t-ffhqu-256x256.pkl", "rb") as f:
        file = pickle.load(f)

    device = torch.device("cpu")
    G = file["G"].to(device)

    # random noise for latent space
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

    # generate image
    raw = G(
        z,
        torch.zeros([1, G.c_dim], device=device),
        truncation_psi=1,
        noise_mode="const",
    )
    img = (raw.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")

    # converting to base64
    im_file = BytesIO()
    img.save(im_file, format="JPEG")

    # returning binary media types
    # https://docs.aws.amazon.com/apigateway/latest/developerguide/lambda-proxy-binary-media.html
    return {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": {"Content-Type": "image/png"},
        "body": base64.b64encode(im_file.getvalue()),
    }

