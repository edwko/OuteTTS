import os
import polars as pl
import torch
from tqdm import tqdm
import outetts

df = pl.read_parquet("sample.parquet")

language = "en"
device = "cuda"

interface = outetts.InterfaceHF(
    model_version="0.2",
    cfg=outetts.HFModelConfig_v1(
        model_path="OuteAI/OuteTTS-0.2-500M",
        language=language,
    )
)

del interface.model

ctc = outetts.CTCForcedAlignment([language], device)

def create_speaker(audio_path: str, transcript: str, language: str):
    words = ctc.align(audio_path, transcript, language)

    full_codes = interface.audio_codec.encode(
        interface.audio_codec.convert_audio_tensor(
            audio=torch.cat([i["audio"] for i in words], dim=1),
            sr=ctc.sample_rate
        ).to(interface.audio_codec.device)
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
    }

data = []

for i in tqdm(df.to_dicts()):
    text = i["text"]
    language = i["language"]

    file = i["audio"]["path"]
    with open(file, 'wb') as f:
        f.write(i["audio"]["bytes"])

    data.append(interface.prompt_processor.get_training_prompt(
        text=text,
        language=language,
        speaker=create_speaker(file, text, language)
    ))

    os.remove(file)

pl.DataFrame({"data": data}).write_parquet("processed_data.parquet")