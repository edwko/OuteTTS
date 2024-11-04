from dataclasses import dataclass
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from llama_cpp import Llama, llama_token_is_eog
    _GGUF_AVAILABLE = True
except ImportError:
    _GGUF_AVAILABLE = False

@dataclass
class GenerationConfig:
    temperature: float = 0.1
    repetition_penalty: float = 1.1
    max_length: int = 4096

class HFModel:
    def __init__(
        self,
        model_path: str,
        device: str = None,
        dtype: torch.dtype = None,
        additional_model_config: dict = {}
    ) -> None:
        self.device = torch.device(
            device if device is not None 
            else "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            **additional_model_config
        ).to(device)

    def generate(self, input_ids: torch.Tensor, config: GenerationConfig) -> list[int]:
        return self.model.generate(
            input_ids,
            max_length=config.max_length,
            temperature=config.temperature,
            repetition_penalty=config.repetition_penalty,
            do_sample=True
        )[0].tolist()

class GGUFModel:
    def __init__(
            self,
            model_path: str,
            n_gpu_layers: int = 0,
            additional_model_config: dict = {}
    ) -> None:
        
        if not _GGUF_AVAILABLE:
            raise ImportError(
                "llama_cpp python module not found."
                "To use the GGUF model you must install llama cpp python manually."
            )

        additional_model_config["n_ctx"] = 4096
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            **additional_model_config
        )
    
    def generate(self, input_ids: list[int], config: GenerationConfig) -> list[int]:
        tokens = []
        for token in self.model.generate(
            input_ids,
            temp=config.temperature,
            repeat_penalty=config.repetition_penalty
        ):
            tokens.append(token)
            if (llama_token_is_eog(self.model._model.model, token) or 
                len(tokens) >= config.max_length):
                break

        return tokens