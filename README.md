# OuteTTS

[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-OuteTTS_0.1_350M-orange)](https://huggingface.co/OuteAI/OuteTTS-0.1-350M)

OuteTTS is an experimental text-to-speech model that uses a pure language modeling approach to generate speech.

## Installation

```bash
pip install outetts
```

**Important:** For GGUF support, you must manually install `llama-cpp-python` first.

Visit https://github.com/abetlen/llama-cpp-python for specific installation instructions

## Usage

### Interface Usage
```python
from outetts.v0_1.interface import InterfaceHF, InterfaceGGUF

# Initialize the interface with the Hugging Face model
interface = InterfaceHF("OuteAI/OuteTTS-0.1-350M")

# Or initialize the interface with a GGUF model
# interface = InterfaceGGUF("path/to/model.gguf")

# Generate TTS output
# Without a speaker reference, the model generates speech with random speaker characteristics
output = interface.generate(
    text="Hello, am I working?",
    temperature=0.1,
    repetition_penalty=1.1,
    max_lenght=4096
)

# Play the generated audio
output.play()

# Save the generated audio to a file
output.save("output.wav")
```

### Voice Cloning
```python
# Create a custom speaker from an audio file
speaker = interface.create_speaker(
    "path/to/reference.wav",
    "reference text matching the audio"
)

# Generate TTS with the custom voice
output = interface.generate(
    text="This is a cloned voice speaking",
    speaker=speaker,
    temperature=0.1,
    repetition_penalty=1.1,
    max_lenght=4096
)
```

## Technical Blog
https://www.outeai.com/blog/OuteTTS-0.1-350M


## Credits

- WavTokenizer: [GitHub Repository](https://github.com/jishengpeng/WavTokenizer)
    - Decoder and encoder folder files are from this repository
- CTC Forced Alignment: [PyTorch Tutorial](https://pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html)
