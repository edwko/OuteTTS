# OuteTTS

[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-OuteTTS_0.2_500M-blue)](https://huggingface.co/OuteAI/OuteTTS-0.2-500M)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-OuteTTS_0.2_500M_GGUF-blue)](https://huggingface.co/OuteAI/OuteTTS-0.2-500M-GGUF)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo_Space-pink)](https://huggingface.co/spaces/OuteAI/OuteTTS-0.2-500M-Demo)
[![PyPI](https://img.shields.io/badge/PyPI-outetts-5c6c7a)](https://pypi.org/project/outetts/)
[![npm](https://img.shields.io/badge/npm-outetts-734440)](https://www.npmjs.com/package/outetts)

🤗 [Hugging Face](https://huggingface.co/OuteAI) | 💬 [Discord](https://discord.gg/vyBM87kAmf) | 𝕏 [X (Twitter)](https://twitter.com/OuteAI) | 🌐 [Website](https://www.outeai.com) | 📰 [Blog](https://www.outeai.com/blog)

OuteTTS is an experimental text-to-speech model that uses a pure language modeling approach to generate speech, without architectural changes to the foundation model itself.

## Compatibility

OuteTTS supports the following backends:

| **Backend**                 | **v0.2 - RTF Score** |
|-----------------------------|-----------------------|
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | ~1.44                  |
| [GGUF llama.cpp](https://github.com/ggerganov/llama.cpp)              | ~0.36                  |
| [ExLlamaV2](https://github.com/turboderp/exllamav2)                   | N/A                   |
| [Transformers.js](https://github.com/huggingface/transformers.js)          | N/A                    |

**Note:** The WavTokenizer and CTC model functionality rely on PyTorch.

## Roadmap

Check out [project roadmap](https://github.com/users/edwko/projects/1) to see what's being worked on and upcoming features.

## Installation

#### Python

```bash
pip install outetts
```

**Important:**
- For GGUF support, install `llama-cpp-python` manually. [Installation Guide](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation)
- For EXL2 support, install `exllamav2` manually. [Installation Guide](https://github.com/turboderp/exllamav2?tab=readme-ov-file#installation)

#### Node.js / Browser

```bash
npm i outetts
```

## Usage

### Quick Start: Basic Full Example

```python
import outetts

# Configure the model
model_config = outetts.HFModelConfig_v1(
    model_path="OuteAI/OuteTTS-0.2-500M",
    language="en",  # Supported languages in v0.2: en, zh, ja, ko
)

# Initialize the interface
interface = outetts.InterfaceHF(model_version="0.2", cfg=model_config)

# Print available default speakers
interface.print_default_speakers()

# Load a default speaker
speaker = interface.load_default_speaker(name="male_1")

# Generate speech
output = interface.generate(
    text="Speech synthesis is the artificial production of human speech.",
    temperature=0.1,
    repetition_penalty=1.1,
    max_length=4096,

    # Optional: Use a speaker profile for consistent voice characteristics
    # Without a speaker profile, the model will generate a voice with random characteristics
    speaker=speaker,
)

# Save the generated speech to a file
output.save("output.wav")

# Optional: Play the generated audio
# output.play()
```

### Backend-Specific Configuration

#### Hugging Face Transformers

```python
import outetts

model_config = outetts.HFModelConfig_v1(
    model_path="OuteAI/OuteTTS-0.2-500M",
    language="en",  # Supported languages in v0.2: en, zh, ja, ko
)

interface = outetts.InterfaceHF(model_version="0.2", cfg=model_config)
```

#### GGUF (llama-cpp-python)

```python
import outetts

model_config = outetts.GGUFModelConfig_v1(
    model_path="local/path/to/model.gguf",
    language="en", # Supported languages in v0.2: en, zh, ja, ko
    n_gpu_layers=0,
)

interface = outetts.InterfaceGGUF(model_version="0.2", cfg=model_config)
```

#### ExLlamaV2

```python
import outetts

model_config = outetts.EXL2ModelConfig_v1(
    model_path="local/path/to/model",
    language="en", # Supported languages in v0.2: en, zh, ja, ko
)

interface = outetts.InterfaceEXL2(model_version="0.2", cfg=model_config)
```

### Speaker Creation and Management

#### Creating a Speaker

You can create a speaker profile for voice cloning, which is compatible across all backends.

```python
speaker = interface.create_speaker(
    audio_path="path/to/audio/file.wav",

    # If transcript is not provided, it will be automatically transcribed using Whisper
    transcript=None,            # Set to None to use Whisper for transcription

    whisper_model="turbo",      # Optional: specify Whisper model (default: "turbo")
    whisper_device=None,        # Optional: specify device for Whisper (default: None)
)
```
#### Saving and Loading Speaker Profiles

Speaker profiles can be saved and loaded across all supported backends.

```python
# Save speaker profile
interface.save_speaker(speaker, "speaker.json")

# Load speaker profile
speaker = interface.load_speaker("speaker.json")
```

#### Default Speaker Initialization

OuteTTS includes a set of default speaker profiles. Use them directly:

```python
# Print available default speakers
interface.print_default_speakers()
# Load a default speaker
speaker = interface.load_default_speaker(name="male_1")
```

### Text-to-Speech Generation

The generation process is consistent across all backends.

```python
output = interface.generate(
    text="Speech synthesis is the artificial production of human speech.",
    temperature=0.1,
    repetition_penalty=1.1,
    max_length=4096,
    speaker=speaker, # Optional: speaker profile
)

output.save("output.wav")
# Optional: Play the audio
# output.play()
```

### Custom Backend Configuration

You can initialize custom backend configurations for specific needs.

#### Example with Flash Attention for Hugging Face Transformers

```python
model_config = outetts.HFModelConfig_v1(
    model_path="OuteAI/OuteTTS-0.2-500M",
    language="en",
    dtype=torch.bfloat16,
    additional_model_config={
        'attn_implementation': "flash_attention_2"
    }
)
```

### Node.js Quick Start

The JavaScript implementation follows the same patterns as the Python version, making it easy to switch between the two.

```javascript
import { HFModelConfig_v1, InterfaceHF } from "outetts";

// Configure the model
const model_config = new HFModelConfig_v1({
    model_path: "onnx-community/OuteTTS-0.2-500M",
    language: "en", // Supported languages in v0.2: en, zh, ja, ko
    dtype: "fp32", // Supported dtypes: fp32, q8, q4
});

// Initialize the interface
const tts_interface = await InterfaceHF({ model_version: "0.2", cfg: model_config });

// Print available default speakers
tts_interface.print_default_speakers();

// Load a default speaker
const speaker = tts_interface.load_default_speaker("male_1");

// Generate speech
const output = await tts_interface.generate({
    text: "Speech synthesis is the artificial production of human speech.",
    temperature: 0.1, // Lower temperature values may result in a more stable tone
    repetition_penalty: 1.1,
    max_length: 4096,

    // Optional: Use a speaker profile for consistent voice characteristics
    // Without a speaker profile, the model will generate a voice with random characteristics
    speaker,
});

// Save the synthesized speech to a file
output.save("output.wav");
```

For browser-based applications, check out the example implementation: https://github.com/huggingface/transformers.js-examples/tree/main/text-to-speech-webgpu

### Speaker Profile Recommendations

To achieve the best results when creating a speaker profile, consider the following recommendations:

1. **Audio Clip Duration:**
   - Use an audio clip of around **10-15 seconds**.
   - This duration provides sufficient data for the model to learn the speaker's characteristics while keeping the input manageable. The model's context length is 4096 tokens, allowing it to generate around 54 seconds of audio in total. However, when a speaker profile is included, this capacity is reduced proportionally to the length of the speaker's audio clip.

2. **Audio Quality:**
   - Ensure the audio is **clear and noise-free**. Background noise or distortions can reduce the model's ability to extract accurate voice features.

3. **Accurate Transcription:**
   - Provide a highly **accurate transcription** of the audio clip. Mismatches between the audio and transcription can lead to suboptimal results.

4. **Speaker Familiarity:**
   - The model performs best with voices that are similar to those seen during training. Using a voice that is **significantly different from typical training samples** (e.g., unique accents, rare vocal characteristics) might result in inaccurate replication.
   - In such cases, you may need to **fine-tune the model** specifically on your target speaker's voice to achieve a better representation.

5. **Parameter Adjustments:**
   - Adjust parameters like `temperature` in the `generate` function to refine the expressive quality and consistency of the synthesized voice.

## Credits

- WavTokenizer: [GitHub Repository](https://github.com/jishengpeng/WavTokenizer)
  - `decoder` and `encoder` folder files are from this repository
- CTC Forced Alignment: [PyTorch Tutorial](https://pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html)
- Uroman: [GitHub Repository](https://github.com/isi-nlp/uroman)
    - "This project uses the universal romanizer software 'uroman' written by Ulf Hermjakob, USC Information Sciences Institute (2015-2020)".
- mecab-python3 [GitHub Repository](https://github.com/SamuraiT/mecab-python3)
