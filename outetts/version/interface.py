import torch
from loguru import logger
import os
import json
import polars as pl
import re

from ..models.config import ModelConfig, GenerationConfig, info
from ..models.hf_model import HFModel
from ..models.gguf_model import GGUFModel
from ..models.exl2_model import EXL2Model, EXL2ModelAsync
from ..models.vllm_model import VLLMModelBatch
from ..models.llamacpp_server import LlamaCPPServerModel, LlamaCPPServerAsyncModel
from ..utils.chunking import chunk_text
from ..utils import preprocessing
from .playback import ModelOutput

class InterfaceHF:
    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        self.config = config
        self.verbose = config.verbose
        if config.interface_version == info.InterfaceVersion.V3:
            self.audio_processor = config.audio_processor(config)
            self.audio_codec = self.audio_processor.audio_codec
        else:
            self.audio_codec = config.audio_codec(config.device, config.audio_codec_path)

        self.prompt_processor = config.prompt_processor(config.tokenizer_path)
        self.model = self.get_model()
        
    def get_model(self):
        return HFModel(
            self.config.model_path, 
            self.config.device, 
            self.config.dtype, 
            self.config.additional_model_config
        )

    def _prepare_prompt(self, prompt: str):
        return self.prompt_processor.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)
    
    def prepare_prompt(self, text: str, speaker: dict = None):
        prompt = self.prompt_processor.get_completion_prompt(text, speaker)
        return self._prepare_prompt(prompt)
    
    def get_audio(self, tokens):
        output = self.prompt_processor.extract_audio_from_tokens(tokens)
        if not output:
            logger.error("No audio tokens found in the output")
            return None

        return self.audio_codec.decode(
            torch.tensor([output], dtype=torch.int64).to(self.audio_codec.device)
        )

    def create_speaker(
            self,
            audio_path: str,
            transcript: str = None,
            whisper_model: str = "turbo",
            whisper_device = None,
        ):
        if transcript and self.config.interface_version == info.InterfaceVersion.V3:
            logger.warning("V3 Interface models do not support speaker creation from transcripts."
                            "Whisper will be used to extract speaker features from the audio file."
                            "This is left for older interface compatibility and may be removed in future versions.")
            
        if self.config.interface_version == info.InterfaceVersion.V3:
            from ..version.v3.utils import create_speaker
            speaker = create_speaker(
                audio_processor=self.audio_processor,
                audio_path=audio_path,
                whisper_model=whisper_model,
                whisper_device=whisper_device
            )
        elif self.config.interface_version == info.InterfaceVersion.V2:
            from ..version.v2.utils import create_speaker
            speaker = create_speaker(
                device=self.config.device,
                audio_codec=self.audio_codec,
                audio_path=audio_path,
                transcript=transcript,
                whisper_model=whisper_model,
                whisper_device=whisper_device
            )
        elif self.config.interface_version == info.InterfaceVersion.V1:
            from ..version.v1.utils import create_speaker
            speaker = create_speaker(
                device=self.config.device,
                audio_codec=self.audio_codec,
                audio_path=audio_path,
                transcript=transcript,
                whisper_model=whisper_model,
                whisper_device=whisper_device
            )
        else:
            raise ValueError(f"Unsupported interface version: {self.config.interface_version}") 
        speaker['interface_version'] = self.config.interface_version.value
        return speaker

    def save_speaker(self, speaker: dict, path: str):
        speaker['interface_version'] = self.config.interface_version.value
        if not path.endswith(".json"):
            path += ".json"
        with open(path, "w") as f: 
            json.dump(speaker, f, indent=2)
        logger.info(f"Speaker saved to: {path}")

    def load_speaker(self, path: str):
        with open(path, "r") as f: 
            speaker = json.load(f)

        version = speaker.get('interface_version', 2)

        # For backwards compatibility
        current_version = self.config.interface_version.value 
        if version == 2 and current_version <= 2:
            return speaker

        if current_version != version:
            raise ValueError(f"Speaker interface version mismatch: {version} != {self.config.interface_version.value}")
        
        return speaker
    
    def decode_and_save_speaker(self, speaker: dict, path: str):
        if self.config.interface_version != info.InterfaceVersion.V3:
            raise ValueError("Speaker decoding is only supported for InterfaceVersion.V3")
        c1, c2 = [], []
        for i in speaker["words"]:
            c1.extend(i["c1"])
            c2.extend(i["c2"])
        ModelOutput(self.audio_codec.decode(
            torch.tensor([[c1,c2]], dtype=torch.int64).to(self.audio_codec.device)
        ), og_sr=self.audio_codec.sr).save(path)

    def print_default_speakers(self):
        if self.config.interface_version == info.InterfaceVersion.V3:
            # _BASE_DIR = os.path.dirname(__file__)
            # _DEFAULT_SPEAKERS_DIR = os.path.join(_BASE_DIR, "v3/default_speakers/parquet/speakers.parquet")
            # if not os.path.exists(_DEFAULT_SPEAKERS_DIR):
            #     raise FileNotFoundError(f"Default speakers file not found: {_DEFAULT_SPEAKERS_DIR}")
            # _DEFAULT_SPEAKERS = {i['speaker']: i for i in pl.read_parquet(_DEFAULT_SPEAKERS_DIR).to_dicts()}
            logger.info(f"Available default speakers v3: {['en-female-1-neutral']}")
        else:
            logger.warning("Default speakers are not supported for this interface version.")

    def load_default_speaker(self, name: str):
        name = name.lower().strip()
        if self.config.interface_version == info.InterfaceVersion.V3:
            _BASE_DIR = os.path.dirname(__file__)
            _DEFAULT_SPEAKER_DIR = os.path.join(_BASE_DIR, "v3/default_speakers/json/en-female-1-neutral.json")
            # _DEFAULT_SPEAKERS_DIR = os.path.join(_BASE_DIR, "v3/default_speakers/parquet/speakers.parquet")
            # _DEFAULT_SPEAKERS_DIR = os.path.join(_BASE_DIR, "v3/default_speakers/parquet/speakers.parquet")
            # if not os.path.exists(_DEFAULT_SPEAKERS_DIR):
            #     raise FileNotFoundError(f"Default speakers file not found: {_DEFAULT_SPEAKERS_DIR}")
            # _DEFAULT_SPEAKERS = {i['speaker']: i for i in pl.read_parquet(_DEFAULT_SPEAKERS_DIR).to_dicts()}
            with open(_DEFAULT_SPEAKER_DIR, "r") as f:
                speaker = json.load(f)
            _DEFAULT_SPEAKERS = {"en-female-1-neutral": speaker}
            if name not in _DEFAULT_SPEAKERS:
                raise ValueError(f"Speaker {name} not found {list(_DEFAULT_SPEAKERS.keys())}")
            return _DEFAULT_SPEAKERS.get(name, {})
        else:
            raise ValueError("Default speakers are not supported for this interface version.")

    def check_generation_max_length(self, max_length):
        if max_length is None:
            raise ValueError("max_length must be specified.")
        if max_length > self.config.max_seq_length:
            raise ValueError(f"Requested max_length ({max_length}) exceeds the current max_seq_length ({self.config.max_seq_length}).")

    def _generate(self, input_ids, config: GenerationConfig):
        output = self.model.generate(
            input_ids=input_ids,
            config=config
        )
        return output[input_ids.size()[-1]:]
    
    def guided_words_generation(self, config: GenerationConfig):
        text_chunks = chunk_text(config.text)
        chunk_size = len(text_chunks)
        word_token = self.prompt_processor.tokenizer.encode(
            self.prompt_processor.special_tokens.word_start, add_special_tokens=False
        )[0]
        end_token = self.prompt_processor.tokenizer.encode(
            self.prompt_processor.special_tokens.audio_end, add_special_tokens=False
        )[0]
        logger.info(f"Created: {chunk_size} text chunks")
        all_outputs = []

        def create_insert(word):
            insert = [word_token] 
            insert.extend(self.prompt_processor.tokenizer.encode(
                word + self.prompt_processor.special_tokens.features, add_special_tokens=False))
            return insert
        
        def cat_inputs(input_ids, output):
            if isinstance(input_ids, torch.Tensor):
                return torch.cat(
                    [input_ids, torch.tensor([output], dtype=torch.int64).to(input_ids.device)], dim=1
                )
            elif isinstance(input_ids, list):
                input_ids.extend(output)
                return input_ids
            elif isinstance(input_ids, str):
                return input_ids + self.prompt_processor.tokenizer.decode(output, skip_special_tokens=False)
            else:
                raise ValueError("Invalid input_ids type")

        for i, chunk in enumerate(text_chunks):
            logger.info(f"Proccessing: Chunk {i+1} / {chunk_size}")
            words = preprocessing.get_words(chunk)
            # Initialize first word
            first_word = words.pop(0)
            print(f"\nInserting first word: {repr(first_word)}")
            input_ids = self.prepare_prompt(chunk + first_word + self.prompt_processor.special_tokens.features, config.speaker)
            output = []
            break_next = False
            while True:
                for token in self.model._generate_stream(input_ids, config):
                    if token == word_token or token == end_token:
                        if not words:
                            break
                        insert = create_insert(words.pop(0))
                        output.extend(insert)
                        all_outputs.extend(insert)
                        print(f"\nInserting: {repr(self.prompt_processor.tokenizer.decode(insert, skip_special_tokens=False))}")
                        input_ids = cat_inputs(input_ids, output)
                        output = []
                        break
                    else:
                        output.append(token)
                        all_outputs.append(token)

                if not words:
                    if break_next:
                        break
                    break_next = True

        return all_outputs

    def chunk_generation(self, config: GenerationConfig):
        text_chunks = chunk_text(config.text)
        audio_chunks = []
        chunk_size = len(text_chunks)

        logger.info(f"Created: {chunk_size} text chunks")
        for i, chunk in enumerate(text_chunks):
            logger.info(f"Processing: Chunk {i+1} / {chunk_size}")

            input_ids = self.prepare_prompt(chunk, config.speaker)

            output = self._generate(input_ids, config)
            audio_chunks.extend(output)

        return audio_chunks

    def stream_generation(self, config: GenerationConfig):
        text_chunks = chunk_text(config.text)
        audio_chunks = []
        chunk_size = len(text_chunks)

        logger.info(f"Created: {chunk_size} text chunks")
        for i, chunk in enumerate(text_chunks):
            logger.info(f"Processing: Chunk {i+1} / {chunk_size}")

            input_ids = self.prepare_prompt(chunk, config.speaker)
            for i in self._generate_stream(input_ids, config):
                yield output # todo: add new class with .text and .audio
    
    def regular_generation(self, config: GenerationConfig):
        input_ids = self.prepare_prompt(config.text, config.speaker)
        return self._generate(input_ids, config)
    
    def batch_generation(self, config: GenerationConfig):
        text_chunks = chunk_text(config.text)
        logger.info(f"Total text batches created: {len(text_chunks)}")

        input_ids = [self.prepare_prompt(i, config.speaker) for i in text_chunks]
        outputs = []

        output = self._generate_batch(input_ids, config)
        outputs.extend([i['text'] for i in output])

        print("\n")
        logger.success("Batch generation finished!")

        c1 = []
        c2 = []
        for i in outputs:
            codebook1 = [int(x) for x in re.findall(r'<\|c1_(\d+)\|>', i)]
            codebook2 = [int(x) for x in re.findall(r'<\|c2_(\d+)\|>', i)]

            if not codebook1 or not codebook2:
                logger.error("Codebook is empty. The audio chunk might be missing segment, which could result in incomplete audio when combined. Skipping...")
                continue
                
            min_len = min(len(codebook1), len(codebook2))
            c1.extend(codebook1[:min_len])
            c2.extend(codebook2[:min_len])
        
        logger.info("Decoding audio")

        outputs = self.audio_codec.decode(
            torch.tensor([[c1,c2]], dtype=torch.int64),
            verbose=True,
            chunk_length=config.dac_decoding_chunk
        )

        return outputs

    def _batch_check(self, config: GenerationConfig):
        if config.generation_type != info.GenerationType.BATCH:
            logger.info("vLLM, EXL2 Async, Llama.cpp async server only support GenerationType.BATCH.")
            logger.info(f"Generation type '{config.generation_type.value}' is not supported. Switching to 'BATCH' mode.")
            return info.GenerationType.BATCH
    
    def generate(self, config: GenerationConfig) -> ModelOutput:
        self.check_generation_max_length(config.max_length)
        if config.text is None:
            raise ValueError("text can not be empty!")

        if self.config.backend in [info.Backend.VLLM, info.Backend.EXL2ASYNC, info.Backend.LLAMACPP_ASYNC_SERVER]:
            _batch_check = self._batch_check(config)
            if _batch_check is not None:
                config.generation_type = _batch_check
        
        if config.generation_type == info.GenerationType.CHUNKED:
            output = self.chunk_generation(config)
        elif config.generation_type == info.GenerationType.STREAM:
            output = self.stream_generation(config)
        elif config.generation_type == info.GenerationType.GUIDED_WORDS:
            logger.warning("Guided words generation is experimental and may not work as expected.")
            if self.config.interface_version != info.InterfaceVersion.V3:
                raise ValueError("Guided words generation is only supported for InterfaceVersion.V3")
            if self.config.backend == info.Backend.HF or self.config.backend == info.Backend.EXL2:
                raise ValueError("Guided words generation supports only llama.cpp backend.")
            output = self.guided_words_generation(config)
        elif config.generation_type == info.GenerationType.REGULAR:
            logger.info("Using regular generation, consider using chunked generation for long texts.")
            output = self.regular_generation(config)
        elif config.generation_type == info.GenerationType.BATCH:
            if self.config.backend == info.Backend.HF or self.config.backend == info.Backend.LLAMACPP or self.config.backend == info.Backend.LLAMACPP_SERVER:
                raise ValueError("Batch generation supports only GenerationType.VLLM, GenerationType.EXL2ASYNC, info.Backend.LLAMACPP_ASYNC_SERVER backends.")
            audio = self.batch_generation(config)
            return ModelOutput(audio, self.audio_codec.sr)
        else:
            raise ValueError(f"Unsupported generation type: {config.generation_type}")
        
        audio = self.get_audio(output)
        return ModelOutput(audio, self.audio_codec.sr)

class InterfaceLLAMACPP(InterfaceHF):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.config = config

    def get_model(self):
        return GGUFModel(
            model_path=self.config.model_path,
            n_gpu_layers=self.config.n_gpu_layers,
            max_seq_length=self.config.max_seq_length,
            additional_model_config=self.config.additional_model_config
        )

    def _generate(self, input_ids, config):
        return self.model.generate(
            input_ids=input_ids,
            config=config,
        )

    def _prepare_prompt(self, prompt: str):
        return self.prompt_processor.tokenizer.encode(prompt, add_special_tokens=False)

class InterfaceEXL2(InterfaceHF):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.config = config

    def get_model(self):
        return EXL2Model(
            model_path=self.config.model_path,
            max_seq_length=self.config.max_seq_length,
            additional_model_config=self.config.additional_model_config,
        )

    def _prepare_prompt(self, prompt: str):
        return prompt
    
    def _generate(self, input_ids, config):
        return self.model.generate(
            input_ids=input_ids,
            config=config,
        )
    
class InterfaceEXL2Async(InterfaceHF):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.config = config

    def get_model(self):
        if "exl2_cache_size" not in self.config.additional_model_config:
            self.config.additional_model_config["exl2_cache_size"] = self.config.max_seq_length * self.config.exl2_cache_seq_multiply
        return EXL2ModelAsync(
            model_path=self.config.model_path,
            max_seq_length=self.config.max_seq_length,
            additional_model_config=self.config.additional_model_config,
        )

    def _prepare_prompt(self, prompt: str):
        return prompt
    
    def _generate_batch(self, input_ids, config):
        return self.model.generate_batch(
            input_ids=input_ids,
            config=config,
        )

class InterfaceVLLMBatch(InterfaceHF):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.config = config

    def get_model(self):
        return VLLMModelBatch(
            model_path=self.config.model_path,
            max_seq_length=self.config.max_seq_length,
            additional_model_config=self.config.additional_model_config,
        )

    def _prepare_prompt(self, prompt: str):
        return prompt
    
    def _generate_batch(self, input_ids, config):
        return self.model.generate_batch(
            input_ids=input_ids,
            config=config,
        )

class InterfaceLlamaCPPServerAsyncBatch(InterfaceHF):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.config = config

    def get_model(self):
        return LlamaCPPServerAsyncModel(
            model_path=self.config.model_path,
            max_seq_length=self.config.max_seq_length,
            additional_model_config=self.config.additional_model_config,
        )

    def _prepare_prompt(self, prompt: str):
        return prompt
    
    def _generate_batch(self, input_ids, config):
        return self.model.generate_batch(
            input_ids=input_ids,
            config=config,
        )

class InterfaceLlamaCPPServer(InterfaceHF):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.config = config

    def get_model(self):
        return LlamaCPPServerModel(
            model_path=self.config.model_path,
            max_seq_length=self.config.max_seq_length,
            additional_model_config=self.config.additional_model_config,
            tokenizer=self.prompt_processor.tokenizer
        )

    def _prepare_prompt(self, prompt: str):
        return prompt
    
    def _generate(self, input_ids, config):
        return self.model.generate(
            input_ids=input_ids,
            config=config,
        )
    