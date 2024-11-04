from transformers import AutoTokenizer
from loguru import logger
import re
import inflect

class PromptProcessor:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.lec = inflect.engine()

        self.bos = "<|im_start|>"
        self.eos = "<|im_end|>"
        self.special_tokens = {
            "audio_code": "<|{}|>",
            "text_start": "<|text_start|>",
            "text_end": "<|text_end|>",
            "audio_start": "<|audio_start|>",
            "audio_end": "<|audio_end|>",
            "time": "<|t_{:.2f}|>",
            "code_start": "<|code_start|>",
            "code_end": "<|code_end|>",
            "text_sep": "<|text_sep|>"
        }
        self.text_prompt = "{bos}\n{text_start}{words}{text_end}\n{audio_start}\n"
        self.map_audio_tokens = self.get_audio_token_map()

    def get_audio_token_map(self) -> dict:
        return {
            self.tokenizer.encode(self.special_tokens["audio_code"].format(i), add_special_tokens=False)[0]: i
            for i in range(4100)
        }   

    def process_text(self, text: str) -> list[str]:
        text = re.sub(r'\d+(\.\d+)?', lambda x: self.lec.number_to_words(x.group()), text.lower())
        text = re.sub(r'[^a-z\s]', '', text)
        return text.split()     

    def create_audio_prompt(self, words: list) -> str:
        prompt = []
        for i in words:
            word = i["word"]
            duration = self.special_tokens["time"].format(i["duration"])
            tokens = "".join([self.special_tokens["audio_code"].format(c) for c in i["codes"]])
            prompt.append(f'{word}{duration}{self.special_tokens["code_start"]}{tokens}{self.special_tokens["code_end"]}')
        return "\n".join(prompt)
        
    def get_completion_prompt(self, text: str, speaker: dict = None) -> str:
        words = self.process_text(text)
        if speaker is not None:
            words = self.process_text(speaker["text"]) + words

        words = f"{self.special_tokens['text_sep']}".join([i.strip() for i in words])

        prompt = self.text_prompt.format(
            bos=self.bos, 
            text_start=self.special_tokens['text_start'], 
            words=words, 
            text_end=self.special_tokens['text_end'],
            audio_start=self.special_tokens['audio_start']
        )

        if speaker is not None:
            prompt += self.create_audio_prompt(speaker["words"])

        return prompt
    
    def extract_audio_from_tokens(self, tokens: list[int]) -> list[int]:
        return [self.map_audio_tokens[i] for i in tokens if i in self.map_audio_tokens]
