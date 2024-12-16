import os
import gradio as gr
import outetts
from outetts.version.v1.interface import _DEFAULT_SPEAKERS

model_config = outetts.HFModelConfig_v1(
    model_path="OuteAI/OuteTTS-0.2-500M",
    language="en",
)
interface = outetts.InterfaceHF(model_version="0.2", cfg=model_config)

def get_available_speakers(language):
    """Get available speakers for the selected language."""
    if language not in interface.languages:
        return []
    speakers = list(_DEFAULT_SPEAKERS[language].keys())
    speakers.insert(0, "None") 
    return speakers

def change_interface_language(language):
    """Change interface language and update available speakers."""
    try:
        interface.change_language(language)
        speakers = get_available_speakers(language)
        return gr.update(choices=speakers, value="male_1"), gr.update(visible=True)
    except ValueError as e:
        return gr.update(choices=["None"], value="None"), gr.update(visible=False)

def generate_tts(
        text, temperature, repetition_penalty, language, 
        speaker_selection, reference_audio, reference_text
    ):
    """Generate TTS with error handling and new features."""
    try:
        # Validate inputs for custom speaker
        if reference_audio and reference_text:
            if not os.path.exists(reference_audio):
                raise ValueError("Reference audio file not found")
            if not reference_text.strip():
                raise ValueError("Reference transcription text is required")
            speaker = interface.create_speaker(reference_audio, reference_text)

        # Use selected default speaker
        elif speaker_selection and speaker_selection != "None":
            speaker = interface.load_default_speaker(speaker_selection)

        # No speaker - random characteristics
        else:
            speaker = None

        # Generate audio
        output = interface.generate(
            text=text,
            speaker=speaker,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_length=4096
        )

        # Verify output
        if output.audio is None:
            raise ValueError("Model failed to generate audio. This may be due to input length constraints or early EOS token.")

        # Save and return output
        output_path = "output.wav"
        output.save(output_path)
        return output_path, None

    except Exception as e:
        return None, str(e)

with gr.Blocks() as demo:
    gr.Markdown("# OuteTTS-0.2-500M Text-to-Speech Demo")

    error_box = gr.Textbox(label="Error Messages", visible=False)

    with gr.Row():
        with gr.Column():
            # Language selection
            language_dropdown = gr.Dropdown(
                choices=list(interface.languages),
                value="en",
                label="Interface Language"
            )

            # Speaker selection
            speaker_dropdown = gr.Dropdown(
                choices=get_available_speakers("en"),
                value="male_1",
                label="Speaker Selection"
            )

            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter text here..."
            )

            temperature = gr.Slider(
                0.1, 1.0,
                value=0.1,
                label="Temperature (lower = more stable tone, higher = more expressive)"
            )

            repetition_penalty = gr.Slider(
                0.5, 2.0,
                value=1.1,
                label="Repetition Penalty"
            )

            gr.Markdown("""
### Voice Cloning Guidelines:
- Use 10-15 seconds of clear, noise-free audio
- Provide accurate transcription
- Longer audio clips will reduce maximum output length
- Custom speaker overrides speaker selection
            """)

            reference_audio = gr.Audio(
                label="Reference Audio (for voice cloning)",
                type="filepath"
            )

            reference_text = gr.Textbox(
                label="Reference Transcription Text",
                placeholder="Enter exact transcription of reference audio"
            )

            submit_button = gr.Button("Generate Speech")

        with gr.Column():
            audio_output = gr.Audio(
                label="Generated Audio",
                type="filepath"
            )

    language_dropdown.change(
        fn=change_interface_language,
        inputs=[language_dropdown],
        outputs=[speaker_dropdown, speaker_dropdown]
    )

    submit_button.click(
        fn=generate_tts,
        inputs=[
            text_input,
            temperature,
            repetition_penalty,
            language_dropdown,
            speaker_dropdown,
            reference_audio,
            reference_text
        ],
        outputs=[audio_output, error_box]
    ).then(
        fn=lambda x: gr.update(visible=bool(x)),
        inputs=[error_box],
        outputs=[error_box]
    )

demo.launch()
