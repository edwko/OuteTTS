import torchaudio
import torch
from .encoder.utils import convert_audio
from .decoder.pretrained import WavTokenizer
import requests
from tqdm import tqdm
import os
import platform
from loguru import logger

class AudioCodec:
    # WavTokenizer implementation: https://github.com/jishengpeng/WavTokenizer

    def __init__(self, device: str = None):
        self.device = torch.device(device if device is not None else "cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = self.get_cache_dir()
        self.model_path = os.path.join(self.cache_dir, "wavtokenizer_large_speech_320_24k.ckpt")
        self.config_path = self.get_config_path()
        self.model_url = "https://huggingface.co/novateur/WavTokenizer-large-speech-75token/resolve/main/wavtokenizer_large_speech_320_24k.ckpt"
        self.ensure_model_exists()
        self.wavtokenizer = WavTokenizer.from_pretrained0802(self.config_path, self.model_path)
        self.wavtokenizer = self.wavtokenizer.to(self.device)
        self.sr = 24000
        self.bandwidth_id = torch.tensor([0])

    def get_config_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "wavtokenizer_config.yaml")

    def get_cache_dir(self):
        return os.path.join(
            os.getenv('APPDATA') if platform.system() == "Windows" else os.path.join(os.path.expanduser("~"), ".cache"), 
            "outeai", "tts", "wavtokenizer_large_speech_75_token")

    def ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            logger.info(f"Downloading WavTokenizer model from {self.model_url}")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.download_file(self.model_path, self.model_url)

    def download_file(self, save_path: str, url: str):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as file, tqdm(
            desc=save_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

    def load_audio(self, path):
        wav, sr = torchaudio.load(path)
        wav = convert_audio(wav, sr, self.sr, 1) 
        return wav.to(self.device)

    def encode(self, audio: torch.Tensor):
        _,discrete_code= self.wavtokenizer.encode_infer(audio, bandwidth_id=torch.tensor([0]).to(self.device))
        return discrete_code
        
    def decode(self, codes):
        features = self.wavtokenizer.codes_to_features(codes)
        audio_out = self.wavtokenizer.decode(features, bandwidth_id=torch.tensor([0]).to(self.device))
        return audio_out

    def save_audio(self, audio: torch.Tensor, path: str):
        torchaudio.save(path, audio.cpu(), sample_rate=self.sr, encoding='PCM_S', bits_per_sample=16)