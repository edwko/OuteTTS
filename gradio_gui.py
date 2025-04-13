import gradio as gr
import outetts
import os
import json
from datetime import datetime
import torch
from pathlib import Path

# Set Hugging Face cache directory to the 'models' folder
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models", "huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), "models", "huggingface", "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.getcwd(), "models", "huggingface", "datasets")

class OuteTTSGUI:
    def __init__(self):
        self.interface = None
        self.current_speaker = None
        self.history = []
        self.models_dir = "models"
        self.custom_speakers_dir = "custom_speakers"
        os.makedirs(self.custom_speakers_dir, exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, "whisper"), exist_ok=True)  # Create whisper directory
        
        # Initialize available models
        self.available_models = {
            "Llama-OuteTTS-1.0-1B": {
                "path": os.path.join(self.models_dir, "Llama-OuteTTS-1.0-1B"),
                "type": "transformers"
            },
            "Llama-OuteTTS-1.0-1B-FP16": {
                "path": os.path.join(self.models_dir, "Llama-OuteTTS-1.0-1B-FP16.gguf"),
                "type": "llamacpp"
            }
        }
        
        # Default speaker profiles
        self.default_speakers = [
            "EN-FEMALE-1-NEUTRAL",
            "EN-FEMALE-2-NEUTRAL",
            "EN-MALE-1-NEUTRAL",
            "EN-MALE-2-NEUTRAL"
        ]
        
        # Create output directory for generated audio
        self.output_dir = "generated_audio"
        os.makedirs(self.output_dir, exist_ok=True)

    def initialize_interface(self, model_name, backend_type, device, dtype_str, additional_model_config_json, use_flash_attention):
        if self.interface is not None:
            del self.interface
        
        model_info = self.available_models[model_name]
        
        try:
            # Parse additional model config
            additional_model_config = json.loads(additional_model_config_json) if additional_model_config_json else {}
        except json.JSONDecodeError as e:
            return f"Error parsing Additional Model Config JSON: {str(e)}"

        # Determine torch dtype
        dtype = getattr(torch, dtype_str, torch.float32) # Default to float32 if invalid

        if backend_type == "Transformers":
            # Add user-defined additional config
            base_additional_config = {}
            if use_flash_attention:
                base_additional_config["attn_implementation"] = "flash_attention_2"
            base_additional_config.update(additional_model_config) 
            
            config = outetts.ModelConfig(
                model_path=model_info["path"],
                tokenizer_path=model_info["path"],
                interface_version=outetts.InterfaceVersion.V3,
                backend=outetts.Backend.HF,
                additional_model_config=base_additional_config,
                device=device,
                dtype=dtype
            )
        else:  # llama.cpp
            # Note: device, dtype, and additional_model_config might be ignored by llama.cpp backend
            # We still pass the model path correctly.
            config = outetts.ModelConfig(
                model_path=model_info["path"], 
                interface_version=outetts.InterfaceVersion.V3,
                backend=outetts.Backend.LLAMACPP,
                # Pass other params, though they might not be used by this backend's manual config
                device=device, 
                dtype=dtype,
                additional_model_config=additional_model_config 
            )
        
        try:
            self.interface = outetts.Interface(config=config)
            return "Model initialized successfully!"
        except Exception as e:
             return f"Error initializing model: {str(e)}"

    def load_speaker(self, speaker_name):
        if self.interface is None:
            return None, "Please initialize the model first!"
        
        try:
            if speaker_name in self.default_speakers:
                self.current_speaker = self.interface.load_default_speaker(speaker_name)
            else:
                speaker_path = os.path.join(self.custom_speakers_dir, speaker_name)
                self.current_speaker = self.interface.load_speaker(speaker_path)
            return None, f"Speaker {speaker_name} loaded successfully!"
        except Exception as e:
            return None, f"Error loading speaker: {str(e)}"

    def create_speaker(self, audio_path, speaker_name, whisper_model, transcript):
        if self.interface is None:
            return None, "Please initialize the model first!"
        
        try:
            if whisper_model == "None":
                speaker = self.interface.create_speaker(audio_path, transcript=transcript)
            else:
                speaker = self.interface.create_speaker(audio_path, whisper_model=whisper_model)
            speaker_path = os.path.join(self.custom_speakers_dir, speaker_name)
            self.interface.save_speaker(speaker, speaker_path)
            return None, f"Speaker {speaker_name} created and saved successfully!"
        except Exception as e:
            return None, f"Error creating speaker: {str(e)}"

    def refresh_speakers(self):
        custom_speakers = [f for f in os.listdir(self.custom_speakers_dir) if f.endswith('.json')]
        all_speakers = self.default_speakers + custom_speakers
        return gr.Dropdown(choices=all_speakers, value=all_speakers[0])

    def generate_speech(self, text, temperature, top_k, top_p, repetition_penalty, min_p,
                          mirostat, mirostat_tau, mirostat_eta, generation_type, max_length,
                          additional_gen_config_json):
        if self.interface is None:
            return None, "Please initialize the model first!"
        if self.current_speaker is None:
            return None, "Please load a speaker first!"
        
        try:
            # Parse additional generation config
            additional_gen_config = json.loads(additional_gen_config_json) if additional_gen_config_json else {}
        except json.JSONDecodeError as e:
            return None, f"Error parsing Additional Gen Config JSON: {str(e)}"

        try:
            sampler_config = outetts.SamplerConfig(
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
                min_p=float(min_p),
                mirostat=mirostat,
                mirostat_tau=float(mirostat_tau),
                mirostat_eta=float(mirostat_eta)
            )

            output = self.interface.generate(
                config=outetts.GenerationConfig(
                    text=text,
                    speaker=self.current_speaker,
                    generation_type=getattr(outetts.GenerationType, generation_type),
                    sampler_config=sampler_config,
                    max_length=int(max_length),
                    additional_gen_config=additional_gen_config # Added additional gen config
                )
            )
            
            # Save the generated audio
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"generated_{timestamp}.wav")
            output.save(output_path)
            
            # Add to history
            self.history.append({
                "timestamp": timestamp,
                "text": text,
                "speaker": self.current_speaker.name if hasattr(self.current_speaker, 'name') else "Unknown",
                "path": output_path
            })
            
            return output_path, "Generation successful!"
        except Exception as e:
            return None, f"Error during generation: {str(e)}"

    def get_history(self):
        if not self.history:
            return "No generation history available."
        
        history_text = "Generation History:\n\n"
        for item in reversed(self.history):
            history_text += f"Time: {item['timestamp']}\n"
            history_text += f"Speaker: {item['speaker']}\n"
            history_text += f"Text: {item['text']}\n"
            history_text += f"File: {item['path']}\n"
            history_text += "-" * 50 + "\n"
        
        return history_text

    def unload_model(self):
        """Unload the current model and free up resources"""
        if self.interface is not None:
            try:
                # Clear the interface
                self.interface = None
                # Clear the current speaker
                self.current_speaker = None
                # Force garbage collection to free up memory
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return "Model unloaded successfully!"
            except Exception as e:
                return f"Error unloading model: {str(e)}"
        else:
            return "No model is currently loaded."

def create_gui():
    gui = OuteTTSGUI()
    
    # Get compatible dtype for default selection
    try:
        from outetts.models.config import get_compatible_dtype
        default_dtype_str = str(get_compatible_dtype()).split('.')[-1]
    except ImportError:
        default_dtype_str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16" if torch.cuda.is_available() else "float32"

    # Get initial speakers list
    custom_speakers = [f for f in os.listdir(gui.custom_speakers_dir) if f.endswith('.json')]
    all_speakers = gui.default_speakers + custom_speakers

    with gr.Blocks(title="OuteTTS GUI") as interface:
        gr.Markdown("# OuteTTS Text-to-Speech Interface")
        
        with gr.Row():
            with gr.Column(scale=2):
                model_dropdown = gr.Dropdown(choices=list(gui.available_models.keys()), label="Select Model", value=list(gui.available_models.keys())[0])
                backend_dropdown = gr.Dropdown(choices=["Transformers", "llama.cpp"], label="Select Backend", value="Transformers")
                
                # --- Model Config Parameters --- 
                with gr.Accordion("Model Configuration", open=False):
                    device = gr.Dropdown(choices=["cuda", "cpu", "auto"], value="cuda" if torch.cuda.is_available() else "cpu", label="Device")
                    dtype_str = gr.Dropdown(choices=["bfloat16", "float16", "float32"], value=default_dtype_str, label="DType")
                    use_flash_attention = gr.Checkbox(value=True, label="Use Flash Attention 2", visible=True)
                    additional_model_config_json = gr.Textbox(label="Additional Model Config (JSON)", placeholder='e.g., { "device_map": "auto" }', lines=2)
                
                with gr.Row():
                    init_button = gr.Button("Initialize Model")
                    unload_button = gr.Button("Unload Model")
                init_output = gr.Textbox(label="Initialization Status", lines=2)

            with gr.Column(scale=1):
                # --- Speaker Selection --- 
                speaker_dropdown = gr.Dropdown(choices=all_speakers, label="Select Speaker", value=all_speakers[0])
                refresh_speaker_button = gr.Button("Refresh Speakers")
                speaker_output = gr.Textbox(label="Speaker Status", lines=2)

                # --- Custom Speaker Creation --- 
                with gr.Accordion("Create Custom Speaker", open=False):
                    audio_input = gr.Audio(label="Reference Audio", type="filepath")
                    speaker_name_input = gr.Textbox(label="Speaker Name", placeholder="Enter a name for the speaker")
                    whisper_model_dropdown = gr.Dropdown(choices=["None"] + os.listdir(os.path.join(gui.models_dir, "whisper")), label="Select Whisper Model")
                    transcript_input = gr.Textbox(label="Transcription", placeholder="Enter transcription if not using Whisper", lines=3)
                    create_speaker_button = gr.Button("Create Speaker")
                    create_speaker_output = gr.Textbox(label="Creation Status", lines=2)

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(label="Input Text", placeholder="Enter text to convert to speech...", lines=5)
                
                # --- Generation Parameters --- 
                with gr.Accordion("Sampler Configuration", open=False):
                    # (Keep existing sliders/checkboxes for sampler config)
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, label="Temperature")
                    repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.1, label="Repetition Penalty")
                    top_k = gr.Slider(minimum=1, maximum=100, value=40, step=1, label="Top K")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, label="Top P")
                    min_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.05, label="Min P")
                    mirostat = gr.Checkbox(value=False, label="Mirostat")
                    mirostat_tau = gr.Slider(minimum=0.0, maximum=10.0, value=5.0, label="Mirostat Tau")
                    mirostat_eta = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, label="Mirostat Eta")
                
                generation_type = gr.Dropdown(choices=["REGULAR", "CHUNKED", "GUIDED_WORDS"], label="Generation Type", value="CHUNKED")
                max_length = gr.Number(value=8192, label="Max Length", precision=0)
                additional_gen_config_json = gr.Textbox(label="Additional Gen Config (JSON)", placeholder='e.g., { "frequency_penalty": 1.0 }', lines=2)

                generate_button = gr.Button("Generate Speech")
            
            with gr.Column(scale=1):
                # --- Output --- 
                audio_output = gr.Audio(label="Generated Audio")
                generation_output = gr.Textbox(label="Generation Status", lines=2)
        
        with gr.Row():
            # --- History --- 
            history_button = gr.Button("Show History")
            history_output = gr.Textbox(label="Generation History", lines=10, interactive=False)
        
        # --- Event Handlers --- 
        init_button.click(
            fn=gui.initialize_interface,
            inputs=[model_dropdown, backend_dropdown, device, dtype_str, additional_model_config_json, use_flash_attention],
            outputs=[init_output]
        )
        
        unload_button.click(
            fn=gui.unload_model,
            inputs=[],
            outputs=[init_output]
        )
        
        refresh_speaker_button.click(
            fn=gui.refresh_speakers,
            inputs=[],
            outputs=[speaker_dropdown]
        )
        
        speaker_dropdown.change(
            fn=gui.load_speaker,
            inputs=[speaker_dropdown],
            outputs=[audio_output, speaker_output] # Clears audio output on speaker load
        )
        
        create_speaker_button.click(
            fn=gui.create_speaker,
            inputs=[audio_input, speaker_name_input, whisper_model_dropdown, transcript_input],
            outputs=[create_speaker_output]
        )
        
        generate_button.click(
            fn=gui.generate_speech,
            inputs=[
                text_input, temperature, top_k, top_p, repetition_penalty, min_p,
                mirostat, mirostat_tau, mirostat_eta, generation_type, max_length,
                additional_gen_config_json
            ],
            outputs=[audio_output, generation_output]
        )
        
        history_button.click(
            fn=gui.get_history,
            inputs=[],
            outputs=[history_output]
        )
    
    return interface

if __name__ == "__main__":
    # Try to import get_compatible_dtype outside the function for default value setting
    try:
        from outetts.models.config import get_compatible_dtype
    except ImportError:
        get_compatible_dtype = None # Define as None if import fails

    interface = create_gui()
    interface.launch(share=False) 