
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
