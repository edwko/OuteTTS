from dataclasses import dataclass, field
import torch
import os
from loguru import logger
from huggingface_hub import hf_hub_download
from pprint import pprint

from . import info
from ..utils import helpers

@dataclass
class SamplerConfig:
    temperature: float = 0.4
    repetition_penalty: float = 1.1
    repetition_range: int = 64
    top_k: int = 40
    top_p: float = 0.9
    min_p: float = 0.05
    mirostat_tau: int = 5
    mirostat_eta: float = 0.1
    mirostat: bool = False

@dataclass
class GenerationConfig:
    text: str
    voice_characteristics: str = None
    speaker: dict = None
    generation_type: info.GenerationType = info.GenerationType.CHUNKED
    sampler_config: SamplerConfig = field(default_factory=SamplerConfig)
    max_length: int = 8192
    additional_gen_config: dict = field(default_factory=lambda: {})
    additional_dynamic_generator_config: dict = field(default_factory=lambda: {})

def get_compatible_dtype():
    """
    Returns the most compatible dtype for PyTorch based on the user's hardware:
    """
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            logger.info("BF16 support available. Using torch.bfloat16 data type.")
            return torch.bfloat16
        else:
            logger.info("BF16 support not available. Using torch.float16 data type.")
            return torch.float16
    else:
        logger.info("CUDA not available. Using torch.float32 data type.")
        return torch.float32
    
class LoadAutoModel:
    def __init__(self):
        self.cache = helpers.get_cache_dir()

    def init_model(
        self, 
        model: info.Models,
        backend: info.Backend,
        quantization: info.LlamaCppQuantization = None,
    ):
        """
        Automatically initializes and returns the path to the specified model.
        """
        model_dir = "OuteAI/" + model.value

        if backend == info.Backend.HF:
            return model_dir  # HF models are handled directly by Hugging Face
        
        elif backend == info.Backend.LLAMACPP:
            if quantization is None:
                raise ValueError("Quantization parameter is required for GGUF models.")
            model_dir += "-GGUF"
            filename = f"{model.value}-{quantization.value}.gguf"
            model_path = hf_hub_download(
                repo_id=model_dir,
                filename=filename,
                local_dir=os.path.join(self.cache, "gguf"),
                local_files_only=False
            )
            return model_path
        
        elif backend == info.Backend.EXL2:
            raise NotImplementedError("Automatic model loading for EXL2 models is not yet supported.")
        
        else:
            raise ValueError(f"Unsupported backend: {backend.value}")

class ModelConfig:
    def __init__(
        self,
        model_path: str = "OuteAI/Llama-OuteTTS-1.0-1B",
        tokenizer_path: str = "OuteAI/Llama-OuteTTS-1.0-1B",
        interface_version: info.InterfaceVersion = info.InterfaceVersion.V3,
        backend: info.Backend = info.Backend.HF,
        verbose: bool = False,
        device: str = None,
        dtype: torch.dtype = None,
        additional_model_config: dict = {},
        audio_codec_path: str = None,
        max_seq_length: int = 8192,
        n_gpu_layers: int = 0,
        exl2_cache_impl = None,
        exl2_async = False,
        **kwargs
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.interface_version = interface_version
        self.backend = backend 
        self.verbose = verbose
        self.device = device
        self.dtype = dtype
        self.additional_model_config = additional_model_config
        self.audio_codec_path = audio_codec_path
        self.max_seq_length = max_seq_length
        self.n_gpu_layers = n_gpu_layers
        self.exl2_cache_impl = exl2_cache_impl
        self.exl2_async = exl2_async

        # Accept and set any extra keyword arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.audio_processor = None
        self.audio_codec = None
        self.prompt_processor = None

        self._init_functions()

    def _init_functions(self):
        if self.interface_version == info.InterfaceVersion.V1:
            from ..version.v1.prompt_processor import PromptProcessor
            from ..wav_tokenizer.audio_codec import AudioCodec
            self.prompt_processor = PromptProcessor
            self.audio_codec = AudioCodec
        elif self.interface_version == info.InterfaceVersion.V2:
            from ..version.v2.prompt_processor import PromptProcessor
            from ..wav_tokenizer.audio_codec import AudioCodec
            self.prompt_processor = PromptProcessor
            self.audio_codec = AudioCodec
        elif self.interface_version == info.InterfaceVersion.V3:
            from ..version.v3.prompt_processor import PromptProcessor
            from ..version.v3.audio_processor import AudioProcessor
            from ..dac.interface import DacInterface
            self.prompt_processor = PromptProcessor
            self.audio_processor = AudioProcessor
            self.audio_codec = DacInterface
        else:
            raise ValueError(f"Unsupported interface version: {self.interface_version}")
    
    @classmethod
    def auto_config(cls, model: info.Models, backend: info.Backend, quantization: info.LlamaCppQuantization = None):
        logger.info(f"Initializing model configuration for {model.value} model with {backend.value} backend.")
        if model != info.Models.VERSION_1_0_SIZE_1B:
            raise ValueError("Only OuteTTS 1.0 model supported for auto configuration.")
        
        model_path = LoadAutoModel().init_model(model, backend, quantization)
        model_info = info.MODEL_INFO[model]

        logger.info(f"Model path: {model_path}")
        config = {
            "model_path": model_path,
            "tokenizer_path": "OuteAI/" + model.value,
            "interface_version": model_info["interface_version"],
            "backend": backend,
            "verbose": False,
            "device": None,
            "dtype": None,
            "additional_model_config": {},
            "audio_codec_path": None,
            "max_seq_length": model_info["max_seq_length"],
            "n_gpu_layers": 0,
        }

        if backend == info.Backend.HF:
            config["dtype"] = get_compatible_dtype()
            try:
                from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
                config["additional_model_config"]["attn_implementation"] = "flash_attention_2"
                logger.success("Flash attention available. Using flash_attention_2 implementation.")
            except:
                logger.warning("Flash attention 2 not available. Using default attention implementation.\nFor faster inference on supported hardware, consider installing FlashAttention using:\npip install flash-attn --no-build-isolation")
        elif backend == info.Backend.LLAMACPP:
            config["n_gpu_layers"] = 99
            logger.info("LLAMA.CPP backend selected. Offloading all layers to GPU.")

        if backend == info.Backend.EXL2:
            raise NotImplementedError("Automatic model loading for EXL2 models is not yet supported.")
        
        logger.info("Using config:")
        pprint(config, indent=2)
        
        return cls(**config)





        
