from loguru import logger
from tqdm import tqdm
from queue import Queue
import threading
import asyncio

from .config import GenerationConfig
from .info import GenerationType

try:
    from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2CacheBase, ExLlamaV2Tokenizer
    from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicGeneratorAsync, ExLlamaV2DynamicJob, ExLlamaV2DynamicJobAsync, ExLlamaV2Sampler
    _EXL2_AVAILABLE = True
except:
    _EXL2_AVAILABLE = False

class EXL2Model:
    def __init__(
            self,
            model_path: str,
            max_seq_length: int = 4096,
            additional_model_config: dict = {},
            cache_impl: ExLlamaV2CacheBase = ExLlamaV2Cache,
    ) -> None:

        if not _EXL2_AVAILABLE:
            raise ImportError(
                "exllamav2 python module not found."
                "To use the EXL2 model you must install exllamav2 manually."
            )
        
        config = ExLlamaV2Config(model_path)
        self.model = ExLlamaV2(config)
        self.cache = cache_impl(self.model, max_seq_len=max_seq_length, lazy=True)
        self.model.load_autosplit(self.cache, progress=True)
        self.tokenizer = ExLlamaV2Tokenizer(config)
        additional_dynamic_generator_config = additional_model_config.get("additional_dynamic_generator_config", {})
        self.generator = ExLlamaV2DynamicGenerator(
            model=self.model, cache=self.cache, tokenizer=self.tokenizer, **additional_dynamic_generator_config
        )

    def generate(self, input_ids: str, config: GenerationConfig):
        if config.generation_type == GenerationType.STREAM:
            return self._generate_stream(input_ids, config)
        return self._generate(input_ids, config)
    
    def _generate_stream(self, input_ids: str, config: GenerationConfig):
        gen_settings = ExLlamaV2Sampler.Settings(
            token_repetition_penalty=config.sampler_config.repetition_penalty,
            temperature=config.sampler_config.temperature,
            token_repetition_range=config.sampler_config.repetition_range,
            top_k=config.sampler_config.top_k,
            top_p=config.sampler_config.top_p,
            min_p=config.sampler_config.min_p,
            mirostat_eta=config.sampler_config.mirostat_eta,
            mirostat_tau=config.sampler_config.mirostat_tau,
            mirostat=config.sampler_config.mirostat,
            **config.additional_gen_config
        )
        tokens = self.tokenizer.encode(input_ids, add_bos=False, add_eos=False, encode_special_tokens=True)
        tokens_size = len(tokens)
        input_size = len(tokens.flatten().tolist())
        job = ExLlamaV2DynamicJob(
            input_ids=tokens,
            max_new_tokens=config.max_length - input_size,
            stop_conditions=[self.tokenizer.eos_token_id],
            gen_settings=gen_settings,
            identifier=1,
            decode_special_tokens=True,
        )
        self.generator.enqueue(job)
        with tqdm(desc="Generating") as pbar:
            while self.generator.num_remaining_jobs() > 0:
                results = self.generator.iterate()
                for result in results:
                    token = result.get("token_ids", None)
                    if token is None:
                        continue
                    pbar.update(1)
                    pbar.set_postfix({"tokens": input_size + tokens_size, "max tokens": config.max_length})
                    for t in token.flatten().tolist():
                        yield t
                        tokens_size += 1

    def _generate(self, input_ids: str, config: GenerationConfig):
        new_tokens = []
        for i in self._generate_stream(input_ids, config):
            new_tokens.append(i)
        return new_tokens

def run_loop(loop):
    loop.run_forever()

class EXL2AsyncLoop:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=run_loop, args=[self.loop], daemon=True)
        self.thread.start()

class AsyncToken:
    def __init__(self, idx, token):
        self.idx = idx
        self.token = token

class AsyncManager:
    def __init__(self, input_size, tokens_size):
        self.input_size = input_size
        self.tokens_size = tokens_size
        self.chunks = []

class EXL2ModelAsync(EXL2Model):
    def __init__(
            self,
            model_path: str,
            max_seq_length: int = 4096,
            additional_model_config: dict = {},
            cache_impl: ExLlamaV2CacheBase = ExLlamaV2Cache,
    ) -> None:

        if not _EXL2_AVAILABLE:
            raise ImportError(
                "exllamav2 python module not found."
                "To use the EXL2 model you must install exllamav2 manually."
            )
        
        config = ExLlamaV2Config(model_path)
        self.model = ExLlamaV2(config)
        self.cache = cache_impl(self.model, max_seq_len=max_seq_length, lazy=True)
        self.model.load_autosplit(self.cache, progress=True)
        self.tokenizer = ExLlamaV2Tokenizer(config)
        self.additional_dynamic_generator_config = additional_model_config.get("additional_dynamic_generator_config", {})
        self.loop = EXL2AsyncLoop()
        asyncio.run_coroutine_threadsafe(self._get_gen(), loop=self.loop.loop).result()

    async def _get_gen(self):
        self.generator = ExLlamaV2DynamicGeneratorAsync(
            model=self.model, cache=self.cache, tokenizer=self.tokenizer, **self.additional_dynamic_generator_config, max_chunk_size=2048
        )

    async def _generate_async_job(self, chunk, idx, pbar, queue, manager, gen_settings, config):
        #print(chunk)
        try:
            job = ExLlamaV2DynamicJobAsync(
                self.generator,
                input_ids=self.tokenizer.encode(chunk, add_bos=False, add_eos=False, encode_special_tokens=True),
                max_new_tokens=2048,
                stop_conditions=[self.tokenizer.eos_token_id],
                gen_settings=gen_settings,
                decode_special_tokens=True,
            ) 
            idx_added = False
            async for result in job:
                if idx_added == False:
                    manager.chunks.append(idx)
                    idx_added = True
                tokens = result.get("token_ids", "")
                if tokens is None:
                    continue
                try:
                    tokens = int(tokens)
                except:
                    continue
                pbar.update(1)
                manager.tokens_size += 1
                pbar.set_postfix({"tokens": manager.input_size + manager.tokens_size, "chunks": manager.chunks})
                queue.put(AsyncToken(idx, tokens))
        except Exception as e:
            print("Exception in engine:")
            print(repr(e))
            import sys, os, traceback
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(traceback.format_exc())
        queue.put(AsyncToken(idx, None))
        manager.chunks.remove(idx)
    
    def _generate_async(self, chunks, original, config: GenerationConfig):
        gen_settings = ExLlamaV2Sampler.Settings(
            token_repetition_penalty=config.sampler_config.repetition_penalty,
            temperature=config.sampler_config.temperature,
            token_repetition_range=config.sampler_config.repetition_range,
            top_k=config.sampler_config.top_k,
            top_p=config.sampler_config.top_p,
            min_p=config.sampler_config.min_p,
            mirostat_eta=config.sampler_config.mirostat_eta,
            mirostat_tau=config.sampler_config.mirostat_tau,
            mirostat=config.sampler_config.mirostat,
            **config.additional_gen_config
        )
        tokens = self.tokenizer.encode(original, add_bos=False, add_eos=False, encode_special_tokens=True)
        tokens_size = len(tokens)
        input_size = len(tokens.flatten().tolist())
        manager = AsyncManager(input_size, tokens_size)
        job_queue = Queue()
        with tqdm(desc="Generating") as pbar:
            for idx, i in enumerate(chunks):
                asyncio.run_coroutine_threadsafe(coro=self._generate_async_job(i, idx, pbar, job_queue, manager, gen_settings, config), loop=self.loop.loop)
            current_chunk = 0
            all_tokens = {idx: list() for idx in range(len(chunks))} # basically just a list of lists but list of lists didnt work right
            while current_chunk < len(chunks):
                token = job_queue.get()
                if token.token == None:
                    current_chunk += 1
                    manager.chunk = current_chunk
                    continue
                all_tokens[token.idx].append(token.token)
            audio = []
            for chunk in [all_tokens[idx] for idx in range(len(chunks))]:
                audio.extend(chunk)
            return audio
