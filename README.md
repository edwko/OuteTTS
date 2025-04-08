## OuteTTS - Unified Text-To-Speech models treating audio as language

🌐 [Website](https://www.outeai.com) | 🤗 [Hugging Face](https://huggingface.co/OuteAI) | 💬 [Discord](https://discord.gg/vyBM87kAmf) | 𝕏 [X (Twitter)](https://twitter.com/OuteAI) | 📰 [Blog](https://www.outeai.com/blog)

[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Llama_OuteTTS_1.0_1B-blue)](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Llama_OuteTTS_1.0_1B_GGUF-blue)](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B-GGUF)
[![PyPI](https://img.shields.io/badge/PyPI-outetts-5c6c7a)](https://pypi.org/project/outetts/)
[![npm](https://img.shields.io/badge/npm-outetts-734440)](https://www.npmjs.com/package/outetts)

## Compatibility  

OuteTTS supports the following backends:  

| **Backend** | **Type** | **Installation** |  
|-----------------------------|---------|----------------------------|  
| [Llama.cpp Python Bindings](https://github.com/abetlen/llama-cpp-python) | Python | ✅ Installed by default |  
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Python | ✅ Installed by default |  
| [ExLlamaV2](https://github.com/turboderp/exllamav2) | Python | ❌ Requires manual installation |  
| [Transformers.js](https://github.com/huggingface/transformers.js) | JavaScript | NPM package |
| [Llama.cpp Directly](https://github.com/ggml-org/llama.cpp/tree/master/examples/tts) | C++ | External library |  

## Installation

### OuteTTS Installation Guide

OuteTTS now installs the llama.cpp Python bindings by default. Therefore, you must specify the installation based on your hardware. For more detailed instructions on building llama.cpp, refer to the following resources: [llama.cpp Build](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) and [llama.cpp Python](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends)

### Pip:

<details>
<summary>Transformers + llama.cpp CPU</summary>

```bash
pip install outetts --upgrade
```
</details>

<details>
<summary>Transformers + llama.cpp CUDA (NVIDIA GPUs)</summary>
For systems with NVIDIA GPUs and CUDA installed:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install outetts --upgrade
```

</details>

<details>
<summary>Transformers + llama.cpp ROCm/HIP (AMD GPUs)</summary>
For systems with AMD GPUs and ROCm (specify your DAMDGPU_TARGETS) installed:

```bash
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install outetts --upgrade
```

</details>

<details>
<summary>Transformers + llama.cpp Vulkan (Cross-platform GPU)</summary>
For systems with Vulkan support:

```bash
CMAKE_ARGS="-DGGML_VULKAN=on" pip install outetts --upgrade
```
</details>

<details>
<summary>Transformers + llama.cpp Metal (Apple Silicon/Mac)</summary>
For macOS systems with Apple Silicon or compatible GPUs:

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install outetts --upgrade
```
</details>

## Usage

### Basic Usage

```python
import outetts

# Initialize the interface
interface = outetts.Interface(
    config=outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_1B,
        # For llama.cpp backend
        backend=outetts.Backend.LLAMACPP,
        quantization=outetts.LlamaCppQuantization.FP16
        # For transformers backend
        # backend=outetts.Backend.HF,
    )
)

# Load the default speaker profile
speaker = interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")

# Or create your own speaker profiles in seconds and reuse them instantly
# speaker = interface.create_speaker("path/to/audio.wav")
# interface.save_speaker(speaker, "speaker.json")
# speaker = interface.load_speaker("speaker.json")

# Generate speech
output = interface.generate(
    config=outetts.GenerationConfig(
        text="Hello, how are you doing?",
        generation_type=outetts.GenerationType.CHUNKED,
        speaker=speaker,
        sampler_config=outetts.SamplerConfig(
            temperature=0.4
        ),
    )
)

# Save to file
output.save("output.wav")
```

## Interface Documentation

For a complete usage guide, refer to the interface documentation here: 

🔗 [interface_usage.md](https://github.com/edwko/OuteTTS/blob/main/docs/interface_usage.md)

## Usage Recommendations for OuteTTS version 1.0
> [!IMPORTANT]
> **Important Sampling Considerations**  
> 
> When using OuteTTS version 1.0, it is crucial to use the settings specified in the [Sampling Configuration](https://github.com/edwko/OuteTTS?tab=readme-ov-file#sampling-configuration) section.
> 
> The **repetition penalty implementation** is particularly important - this model requires penalization applied to a **64-token recent window**,
> rather than across the entire context window. Penalizing the entire context will cause the model to produce **broken or low-quality output**.
> 
> Currently, **llama.cpp** delivers the most reliable and consistent output quality by default.
> Both **llama.cpp** and **EXL2** support this windowed sampling approach, while **Transformers** doesn't.
> 
> To address this limitation, I've implemented a **windowed repetition penalty** for the **Hugging Face Transformers** backend in the **OuteTTS** library,
> which significantly improves output quality and resolves sampling issues, providing comparable results to llama.cpp.

### Speaker Reference
The model is designed to be used with a speaker reference. Without one, it generates random vocal characteristics, often leading to lower-quality outputs. 
The model inherits the referenced speaker's emotion, style, and accent. 
Therefore, when transcribing to other languages with the same speaker, you may observe the model retaining the original accent. 
For example, if you use a Japanese speaker and continue speech in English, the model may tend to use a Japanese accent.

### Multilingual Application
It is recommended to create a speaker profile in the language you intend to use. This helps achieve the best results in that specific language, including tone, accent, and linguistic features.

While the model supports cross-lingual speech, it still relies on the reference speaker. If the speaker has a distinct accent—such as British English—other languages may carry that accent as well.

### Optimal Audio Length
- **Best Performance:** Generate audio around **42 seconds** in a single run (approximately 8,192 tokens). It is recomended not to near the limits of this windows when generating. Usually, the best results are up to 7,000 tokens.
- **Context Reduction with Speaker Reference:** If the speaker reference is 10 seconds long, the effective context is reduced to approximately 32 seconds.

### Temperature Setting Recommendations
Testing shows that a temperature of **0.4** is an ideal starting point for accuracy (with the sampling settings below). However, some voice references may benefit from higher temperatures for enhanced expressiveness or slightly lower temperatures for more precise voice replication.

### Verifying Speaker Encoding
If the cloned voice quality is subpar, check the encoded speaker sample. 

```python
interface.decode_and_save_speaker(speaker=your_speaker, path="speaker.wav")
```

The DAC audio reconstruction model is lossy, and samples with clipping, excessive loudness, or unusual vocal features may introduce encoding issues that impact output quality.

### Sampling Configuration
For optimal results with this TTS model, use the following sampling settings.

| Parameter         | Value    |
|-------------------|----------|
| Temperature       | 0.4      |
| Repetition Penalty| 1.1      |
| **Repetition Range**  | **64**       |
| Top-k             | 40       |
| Top-p             | 0.9      |
| Min-p             | 0.05     |
