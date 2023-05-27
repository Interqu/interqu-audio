# imports
import torch
import torch.nn as nn
import numpy as np
import os
import librosa
import json
from sklearn.preprocessing import StandardScaler
import boto3
import soundfile as sf

# define client session
s3_client = boto3.client("s3")

EMOTIONS = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fear",
    7: "disgust",
    0: "surprise",
}
DATA_PATH = "/tmp/splits/"
SAMPLE_RATE = 48000


def getMELspectrogram(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        win_length=512,
        window="hamming",
        hop_length=256,
        n_mels=128,
        fmax=sample_rate / 2,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


# model definition
class ParallelModel(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()
        # conv block
        self.conv2Dblock = nn.Sequential(
            # 1. conv block
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            # 2. conv block
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            # 3. conv block
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            # 4. conv block
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        # LSTM block
        self.lstm_maxpool = nn.MaxPool2d(kernel_size=[2, 4], stride=[2, 4])
        hidden_size = 128
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=hidden_size, bidirectional=True, batch_first=True
        )
        self.dropout_lstm = nn.Dropout(0.1)
        self.attention_linear = nn.Linear(
            2 * hidden_size, 1
        )  # 2*hidden_size for the 2 outputs of bidir LSTM
        # Linear softmax layer
        self.out_linear = nn.Linear(2 * hidden_size + 256, num_emotions)
        self.dropout_linear = nn.Dropout(p=0)
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # conv embedding
        conv_embedding = self.conv2Dblock(x)  # (b,channel,freq,time)
        conv_embedding = torch.flatten(
            conv_embedding, start_dim=1
        )  # do not flatten batch dimension
        # lstm embedding
        x_reduced = self.lstm_maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(0, 2, 1)  # (b,t,freq)
        lstm_embedding, (h, c) = self.lstm(x_reduced)  # (b, time, hidden_size*2)
        lstm_embedding = self.dropout_lstm(lstm_embedding)
        batch_size, T, _ = lstm_embedding.shape
        attention_weights = [None] * T
        for t in range(T):
            embedding = lstm_embedding[:, t, :]
            attention_weights[t] = self.attention_linear(embedding)
        attention_weights_norm = nn.functional.softmax(
            torch.stack(attention_weights, -1), -1
        )
        attention = torch.bmm(
            attention_weights_norm, lstm_embedding
        )  # (Bx1xT)*(B,T,hidden_size*2)=(B,1,2*hidden_size)
        attention = torch.squeeze(attention, 1)
        # concatenate
        complete_embedding = torch.cat([conv_embedding, attention], dim=1)

        output_logits = self.out_linear(complete_embedding)
        output_logits = self.dropout_linear(output_logits)
        output_softmax = self.out_softmax(output_logits)

        return output_logits, output_softmax, attention_weights_norm


def generateInputTensor():
    # load split files
    signals = []

    # loading and signals
    for wav_path in os.listdir(DATA_PATH):
        audio, sample_rate = librosa.load(
            f"{DATA_PATH}{wav_path}", duration=3, offset=0.5, sr=SAMPLE_RATE
        )
        signal = np.zeros(
            (
                int(
                    SAMPLE_RATE * 3,
                )
            )
        )
        signal[: len(audio)] = audio
        signals.append(signal)
    signals = np.stack(signals, axis=0)  # fails if no signals are in here

    scaler = StandardScaler()

    X = signals
    input = []
    input.append(X[0:])
    input = np.concatenate(input, 0)

    # spectrograms
    mel_test = []
    print("Calculatin mel spectrograms for test set")
    for i in range(input.shape[0]):
        mel_spectrogram = getMELspectrogram(input[i, :], sample_rate=SAMPLE_RATE)
        mel_test.append(mel_spectrogram)
    mel_test = np.stack(mel_test, axis=0)
    del input
    input = mel_test

    input = np.expand_dims(input, 1)
    b, c, h, w = input.shape
    input = np.reshape(input, newshape=(b, -1))
    input = scaler.fit_transform(input)
    input = np.reshape(input, newshape=(b, c, h, w))

    return torch.tensor(input, device="cpu").float()


def split_audio(audio_path):
    # split into 3 second lengths
    os.makedirs("/tmp/splits/", exist_ok=True)
    wave, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # segment duration
    seg_duration = 3  # seconds
    seg_length = sr * seg_duration

    # num sections
    num_sections = int(np.ceil(len(wave) / seg_length))
    splits = []

    for i in range(num_sections):
        t = wave[i * seg_length : (i + 1) * seg_length]
        splits.append(t)

    for i in range(num_sections):
        recording_name = os.path.basename("audio"[:-4])
        out_file = f"{recording_name}_{str(i)}.wav"
        sf.write(os.path.join(DATA_PATH, out_file), splits[i], sr)

    print(f"split audio using sample rate to be: {sr}")


def lambda_handler(event, context):
    # determine what audio file to fetch
    audio_file = event["queryStringParameters"]["file_name"]
    if not audio_file:
        return {
            "statusCode": 400,
            "body": json.dumps(
                {
                    "error": "no file name was provided",
                }
            ),
        }

    # load saved model
    LOAD_PATH = os.path.join(os.getcwd(), "models")
    print(LOAD_PATH)
    model = ParallelModel(len(EMOTIONS))
    model.load_state_dict(
        torch.load("/opt/ml/cnn_model.pt", map_location=torch.device("cpu"))
    )
    print(
        "Model is loaded from {}".format(
            os.path.join(LOAD_PATH, "cnn_lstm_parallel_model.pt")
        )
    )

    # download the audio file
    s3_client.download_file("interqu-audio", audio_file, "/tmp/audio.wav")

    # split the audio file
    split_audio("/tmp/audio.wav")

    # load the audio files
    input_tensor = generateInputTensor()

    # make predictions
    with torch.no_grad():
        model.eval()
        output_logits, output_softmax, attention_weights_norm = model(input_tensor)
        predictions = torch.argmax(output_softmax, dim=1).tolist()

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "predictions": predictions,
            }
        ),
    }
