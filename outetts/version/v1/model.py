from dataclasses import dataclass, field
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from llama_cpp import Llama, llama_token_is_eog
    _GGUF_AVAILABLE = True
except ImportError:
    _GGUF_AVAILABLE = False

try:
    from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
    from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
    _EXL2_AVAILABLE = True
except ImportError:
    _EXL2_AVAILABLE = False

@dataclass
class GenerationConfig:
    temperature: float = 0.1
    repetition_penalty: float = 1.1
    max_length: int = 4096
    additional_gen_config: dict = field(default_factory=lambda: {})

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

    def generate(self, input_ids: torch.Tensor, config: GenerationConfig, stream: bool = False) -> list[int]:
        if stream:
            raise NotImplementedError("Stream generation is not supported for HF models.")

        return self.model.generate(
            input_ids,
            max_length=config.max_length,
            temperature=config.temperature,
            repetition_penalty=config.repetition_penalty,
            do_sample=True,
            **config.additional_gen_config,
        )[0].tolist()

class GGUFModel:
    def __init__(
            self,
            model_path: str,
            n_gpu_layers: int = 0,
            max_seq_length: int = 4096,
            additional_model_config: dict = {}
    ) -> None:

        if not _GGUF_AVAILABLE:
            raise ImportError(
                "llama_cpp python module not found."
                "To use the GGUF model you must install llama cpp python manually."
            )

        additional_model_config["n_ctx"] = max_seq_length
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            **additional_model_config
        )

    def generate(self, input_ids: list[int], config: GenerationConfig, stream: bool = False):
        if stream:
            return self._generate_stream(input_ids, config)
        return self._generate(input_ids, config)

    def _generate_stream(self, input_ids: list[int], config: GenerationConfig):
        for token in self.model.generate(
            input_ids,
            temp=config.temperature,
            repeat_penalty=config.repetition_penalty,
            **config.additional_gen_config,
        ):
            yield token
            if llama_token_is_eog(self.model._model.model, token):
                break

    def _generate(self, input_ids: list[int], config: GenerationConfig) -> list:
        tokens = []
        for token in self.model.generate(
            input_ids,
            temp=config.temperature,
            repeat_penalty=config.repetition_penalty,
            **config.additional_gen_config,
        ):
            tokens.append(token)
            if (llama_token_is_eog(self.model._model.model, token) or 
                len(tokens) >= config.max_length):
                break
        return tokens

class EXL2Model:
    def __init__(
            self,
            model_path: str,
            max_seq_length: int,
            additional_model_config: dict = {},
    ) -> None:

        if not _EXL2_AVAILABLE:
            raise ImportError(
                "exllamav2 python module not found."
                "To use the EXL2 model you must install exllamav2 manually."
            )

        config = ExLlamaV2Config(model_path)
        config.arch_compat_overrides()
        self.model = ExLlamaV2(config)
        self.cache = ExLlamaV2Cache(self.model, max_seq_len=config.max_seq_len, lazy=True)
        self.model.load_autosplit(self.cache, progress=True)
        self.tokenizer = ExLlamaV2Tokenizer(config)

    def generate(self, input_ids: list[int], config: GenerationConfig,, additional_dynamic_generator_config: dict, stream: bool = False):
        if stream:
            return self._generate_stream(input_ids, config, additional_dynamic_generator_config)
        return self._generate(input_ids, config, additional_dynamic_generator_config)

    def _generate(self, input_ids: str, config: GenerationConfig, additional_dynamic_generator_config: dict) -> list[int]:
        generator = ExLlamaV2DynamicGenerator(
            model = self.model,
            cache = self.cache,
            tokenizer = self.tokenizer,
            **additional_dynamic_generator_config
        )
        if stream:
            raise NotImplementedError("Stream generation is not supported for EXL2 models.")

        gen_settings = ExLlamaV2Sampler.Settings(
                    token_repetition_penalty=config.repetition_penalty,
                    temperature=config.temperature,
                    **config.additional_gen_config
                ),

        input_size = self.tokenizer.encode(input_ids).size()[-1]

        output = generator.generate(
            input_ids = input_ids,
            max_new_tokens = config.max_length,
            add_bos = False,
            decode_special_tokens=True,
            gen_settings = gen_settings,
            stop_conditions=[self.tokenizer.eos_token_id]
        )

        return self.tokenizer.encode(output).flatten().tolist()[input_size:]

    
    def _generate_stream(self, input_ids: list[int], config: GenerationConfig, additional_dynamic_generator_config: dict):
        generator = ExLlamaV2DynamicGenerator(
            model = self.model,
            cache = self.cache,
            tokenizer = self.tokenizer,
            **additional_dynamic_generator_config,
        )
        job = ExLlamaV2DynamicJob(
            input_ids=input_ids.to("cpu"),
            max_new_tokens=config.max_length,
            gen_settings = ExLlamaV2Sampler.Settings(token_repetition_penalty=config.repetition_penalty, temperature=config.temperature, **config.additional_gen_config),
        )
        generator.enqueue(job)
        eos = False
        while not eos:
            results = generator.iterate()
            # Only batches one at a time. Has room for improvement.
            for result in results:
                assert result["job"] == job
                if result["stage"] == "streaming":
                    token = int(result.get("token_ids", "")[0][0])
                    yield token
                    if token == self.tokenizer.eos_token_id:
                        eos = True
