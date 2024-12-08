import json
import os
from dataclasses import dataclass, field
from typing import Generator

import torch
import torchaudio
from loguru import logger

from ...wav_tokenizer.audio_codec import AudioCodec
from ...whisper import transcribe
from .alignment import CTCForcedAlignment
from .model import EXL2Model, GenerationConfig, GGUFModel, HFModel
from .prompt_processor import PromptProcessor

try:
    import sounddevice as sd
    ENABLE_PLAYBACK = True
except Exception as e:
    ENABLE_PLAYBACK = False
    logger.error(e)
    logger.warning("Failed to import sounddevice. Audio playback is disabled.")

BASE_DIR = os.path.dirname(__file__)

DEFAULT_SPEAKERS_DIR = os.path.join(BASE_DIR, "default_speakers")

def get_speaker_path(speaker_name):
    return os.path.join(DEFAULT_SPEAKERS_DIR, f"{speaker_name}.json")

_DEFAULT_SPEAKERS = {
    "en": {
        "male_1": get_speaker_path("en_male_1"),
        "male_2": get_speaker_path("en_male_2"), 
        "male_3": get_speaker_path("en_male_3"),
        "male_4": get_speaker_path("en_male_4"),
        "female_1": get_speaker_path("en_female_1"),
        "female_2": get_speaker_path("en_female_2"),
    },
    "ja": {
        "male_1": get_speaker_path("ja_male_1"),
        "female_1": get_speaker_path("ja_female_1"),
        "female_2": get_speaker_path("ja_female_2"),
        "female_3": get_speaker_path("ja_female_3"),
    },
    "ko": {
        "male_1": get_speaker_path("ko_male_1"),
        "male_2": get_speaker_path("ko_male_2"),
        "female_1": get_speaker_path("ko_female_1"),
        "female_2": get_speaker_path("ko_female_2"),
    },
    "zh": {
        "male_1": get_speaker_path("zh_male_1"),
        "female_1": get_speaker_path("zh_female_1"),
    }
}

@dataclass
class HFModelConfig:
    model_path: str = "OuteAI/OuteTTS-0.2-500M"
    language: str = "en"
    tokenizer_path: str = None
    languages: list = field(default_factory=list)
    verbose: bool = False
    device: str = None
    dtype: torch.dtype = None
    additional_model_config: dict = field(default_factory=dict)
    wavtokenizer_model_path: str = None
    max_seq_length: int = 4096

@dataclass
class GGUFModelConfig(HFModelConfig):
    n_gpu_layers: int = 0

@dataclass
class EXL2ModelConfig(HFModelConfig):
    pass

@dataclass
class ModelOutput:
    audio: torch.Tensor
    sr: int
    enable_playback: bool = ENABLE_PLAYBACK

    def save(self, path: str):
        if self.audio is None:
            logger.warning("Audio is empty, skipping save.")
            return

        torchaudio.save(path, self.audio.cpu(), sample_rate=self.sr, encoding='PCM_S', bits_per_sample=16)

    def play(self):
        if self.audio is None:
            logger.warning("Audio is empty, skipping playback.")
            return
        
        if not self.enable_playback:
            logger.warning("Audio playback is disabled. Check sounddevice installation.")
            return

        try:
            sd.play(self.audio[0].cpu().numpy(), self.sr)
            sd.wait()
        except Exception as e:
            logger.error(e)

class InterfaceHF:
    def __init__(
        self,
        config: HFModelConfig
    ) -> None:
        self.device = torch.device(
            config.device if config.device is not None
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.config = config
        self._device = config.device
        self.languages = config.languages
        self.language = config.language
        self.verbose = config.verbose

        self.audio_codec = AudioCodec(self.device, config.wavtokenizer_model_path)
        self.prompt_processor = PromptProcessor(config.tokenizer_path, self.languages)
        self.model = HFModel(config.model_path, self.device, config.dtype, config.additional_model_config)

    def prepare_prompt(self, text: str, speaker: dict = None):
        prompt = self.prompt_processor.get_completion_prompt(text, self.language, speaker)
        return self.prompt_processor.tokenizer.encode(
            prompt, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).to(self.model.device)

    def get_audio(self, tokens):
        output = self.prompt_processor.extract_audio_from_tokens(tokens)
        if not output:
            logger.warning("No audio tokens found in the output")
            return None

        return self.audio_codec.decode(
            torch.tensor([[output]], dtype=torch.int64).to(self.audio_codec.device)
        )

    def create_speaker(
            self, 
            audio_path: str, 
            transcript: str = None, 
            whisper_model: str = "turbo",
            whisper_device = None
        ):

        if transcript is None:
            logger.info("Transcription not provided, transcribing audio with whisper.")
            transcript = transcribe.transcribe_once(
                audio_path=audio_path,
                model=whisper_model,
                device=whisper_device
            )

        if not transcript:
            raise ValueError("Transcript text is empty")

        ctc = CTCForcedAlignment(self.languages, self._device)
        words = ctc.align(audio_path, transcript, self.language)
        ctc.free()

        full_codes = self.audio_codec.encode(
            self.audio_codec.convert_audio_tensor(
                audio=torch.cat([i["audio"] for i in words], dim=1),
                sr=ctc.sample_rate
            ).to(self.audio_codec.device)
        ).tolist()

        data = []
        start = 0
        for i in words:
            end = int(round((i["x1"] / ctc.sample_rate) * 75))
            word_tokens = full_codes[0][0][start:end]
            start = end
            if not word_tokens:
                word_tokens = [1]

            data.append({
                "word": i["word"],
                "duration": round(len(word_tokens) / 75, 2),
                "codes": word_tokens
            })

        return {
            "text": transcript,
            "words": data,
            "language": self.language
        }

    def save_speaker(self, speaker: dict, path: str):
        with open(path, "w") as f:
            json.dump(speaker, f, indent=2)

    def load_speaker(self, path: str):
        with open(path, "r") as f:
            return json.load(f)
        
    def print_default_speakers(self):
        total_speakers = sum(len(speakers) for speakers in _DEFAULT_SPEAKERS.values())
        print("\n=== ALL AVAILABLE SPEAKERS ===")
        print(f"Total: {total_speakers} speakers across {len(_DEFAULT_SPEAKERS)} languages")
        print("-" * 50)
        
        for language, speakers in _DEFAULT_SPEAKERS.items():
            print(f"\n{language.upper()} ({len(speakers)} speakers):")
            for speaker_name in speakers.keys():
                print(f"  - {speaker_name}")
        
        print("\n\n=== SPEAKERS FOR CURRENT INTERFACE LANGUAGE ===")
        if self.language.lower() in _DEFAULT_SPEAKERS:
            current_speakers = _DEFAULT_SPEAKERS[self.language.lower()]
            print(f"Language: {self.language.upper()} ({len(current_speakers)} speakers)")
            print("-" * 50)
            for speaker_name in current_speakers.keys():
                print(f"  - {speaker_name}")
        else:
            print(f"No speakers available for current language: {self.language}")
        
        print("\nTo use a speaker: load_default_speaker(name)\n")
        
    def load_default_speaker(self, name: str):
        name = name.lower().strip()
        language = self.language.lower().strip()
        if language not in _DEFAULT_SPEAKERS:
            raise ValueError(f"Speaker for language {language} not found")
        
        speakers = _DEFAULT_SPEAKERS[language]
        if name not in speakers:
            raise ValueError(f"Speaker {name} not found for language {language}")
        
        return self.load_speaker(speakers[name])
    
    def change_language(self, language: str):
        language = language.lower().strip()
        if language not in self.languages:
            raise ValueError(f"Language {language} is not supported by the current model")
        self.language = language

    def check_generation_max_length(self, max_length):
        if max_length is None:
            raise ValueError("max_length must be specified.")
        if max_length > self.config.max_seq_length:
            raise ValueError(f"Requested max_length ({max_length}) exceeds the current max_seq_length ({self.config.max_seq_length}).")

    def generate(
            self, 
            text: str, 
            speaker: dict = None, 
            temperature: float = 0.1, 
            repetition_penalty: float = 1.1, 
            max_length: int = 4096,
            additional_gen_config={},
        ) -> ModelOutput:
        input_ids = self.prepare_prompt(text, speaker)
        if self.verbose:
            logger.info(f"Input tokens: {input_ids.size()[-1]}")
            logger.info("Generating audio...")

        self.check_generation_max_length(max_length)

        output = self.model.generate(
            input_ids=input_ids,
            config=GenerationConfig(
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_length=max_length,
                additional_gen_config=additional_gen_config,
            )
        )
        audio = self.get_audio(output[input_ids.size()[-1]:])
        if self.verbose:
            logger.info("Audio generation completed")

        return ModelOutput(audio, self.audio_codec.sr)

class InterfaceGGUF(InterfaceHF):
    def __init__(
        self,
        config: GGUFModelConfig
    ) -> None:
        self.device = torch.device(
            config.device if config.device is not None
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.config = config
        self._device = config.device
        self.languages = config.languages
        self.language = config.language
        self.verbose = config.verbose

        self.audio_codec = AudioCodec(self.device, config.wavtokenizer_model_path)
        self.prompt_processor = PromptProcessor(config.tokenizer_path, self.languages)
        self.model = GGUFModel(
            model_path=config.model_path,
            n_gpu_layers=config.n_gpu_layers,
            max_seq_length=config.max_seq_length,
            additional_model_config=config.additional_model_config
        )

    def prepare_prompt(self, text: str, speaker: dict = None):
        prompt = self.prompt_processor.get_completion_prompt(text, self.language, speaker)
        return self.prompt_processor.tokenizer.encode(prompt, add_special_tokens=False)

    def generate(
            self, 
            text: str, 
            speaker: dict = None, 
            temperature: float = 0.1, 
            repetition_penalty: float = 1.1,
            max_length = 4096,
            additional_gen_config = {},
        ) -> ModelOutput:
        input_ids = self.prepare_prompt(text, speaker)
        if self.verbose:
            logger.info(f"Input tokens: {len(input_ids)}")
            logger.info("Generating audio...")
        
        self.check_generation_max_length(max_length)
        
        output = self.model.generate(
            input_ids=input_ids,
            config=GenerationConfig(
                temperature=temperature,
                max_length=max_length,
                repetition_penalty=repetition_penalty,
                additional_gen_config=additional_gen_config,
            )
        )
        audio = self.get_audio(output)
        if self.verbose:
            logger.info("Audio generation completed")

        return ModelOutput(audio, self.audio_codec.sr)
    
    def generate_stream(
            self, 
            text: str, 
            speaker: dict = None, 
            temperature: float = 0.1, 
            repetition_penalty: float = 1.1,
            max_length = 4096,
            chunk_size = 50,
            additional_gen_config = {},
    ) -> Generator[ModelOutput, None, None]:
        """
        Generate audio tokens in a streaming manner.
        
        :param text: Input text to generate audio for
        :param speaker: Optional speaker information
        :param temperature: Sampling temperature
        :param repetition_penalty: Penalty for token repetition
        :param max_length: Maximum number of tokens to generate
        :param additional_gen_config: Additional generation configurations
        :param chunk_size: Number of tokens to generate per chunk
        :yield: Incremental ModelOutput with audio chunks
        """
        input_ids = self.prepare_prompt(text, speaker)
        if self.verbose:
            logger.info(f"Input tokens: {len(input_ids)}")
            logger.info("Streaming audio generation...")
        
        self.check_generation_max_length(max_length)
        
        # Track tokens for progressive audio generation
        generated_tokens = []
        
        # Stream generation
        for token in self.model.generate_stream(
            input_ids=input_ids,
            config=GenerationConfig(
                temperature=temperature,
                max_length=max_length,
                repetition_penalty=repetition_penalty,
                additional_gen_config=additional_gen_config,
            )
        ):
            generated_tokens.append(token)
            
            # Periodically convert tokens to audio chunks
            # You might want to adjust the chunk size based on your specific requirements
            if len(generated_tokens) % chunk_size == 0:  # Example: generate chunk every 50 tokens
                try:
                    audio_chunk = self.get_audio(generated_tokens)
                    yield ModelOutput(audio_chunk, self.audio_codec.sr)
                    generated_tokens = []
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"Error generating audio chunk: {e}")
        
        # Final audio chunk
        if generated_tokens:
            final_audio = self.get_audio(generated_tokens)
            yield ModelOutput(final_audio, self.audio_codec.sr)
        
        if self.verbose:
            logger.info("Streaming audio generation completed")

class InterfaceEXL2(InterfaceHF):
    def __init__(
        self,
        config: EXL2ModelConfig
    ) -> None:
        self.device = torch.device(
            config.device if config.device is not None
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.config = config
        self._device = config.device
        self.languages = config.languages
        self.language = config.language
        self.verbose = config.verbose

        self.audio_codec = AudioCodec(self.device, config.wavtokenizer_model_path)
        self.prompt_processor = PromptProcessor(config.tokenizer_path, self.languages)
        self.model = EXL2Model(
            model_path=config.model_path,
            max_seq_length=config.max_seq_length,
            additional_model_config=config.additional_model_config,
        )

    def prepare_prompt(self, text: str, speaker: dict = None):
        return self.prompt_processor.get_completion_prompt(text, self.language, speaker)

    def generate(
            self, 
            text: str, 
            speaker: dict = None, 
            temperature: float = 0.1, 
            repetition_penalty: float = 1.1,
            max_length = 4096,
            additional_gen_config = {},
            additional_dynamic_generator_config = {},
        ) -> ModelOutput:
        input_ids = self.prepare_prompt(text, speaker)
        if self.verbose:
            logger.info(f"Input tokens: {len(input_ids)}")
            logger.info("Generating audio...")

        self.check_generation_max_length(max_length)
        
        output = self.model.generate(
            input_ids=input_ids,
            config=GenerationConfig(
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_length=max_length,
                additional_gen_config=additional_gen_config,
            ),
            additional_dynamic_generator_config=additional_dynamic_generator_config
        )
        audio = self.get_audio(output)
        if self.verbose:
            logger.info("Audio generation completed")

        return ModelOutput(audio, self.audio_codec.sr)
