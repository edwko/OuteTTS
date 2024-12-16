import { AutoModelForCausalLM, AutoTokenizer, PreTrainedModel } from "@huggingface/transformers";

import _DEFAULT_SPEAKERS from "./default_speakers.js";
import { PromptProcessor } from "./prompt_processor.js";
import { AudioCodec } from "./audio_codec.js";
import { HFModel } from "./model.js";

export class HFModelConfig {
    constructor({
        model_path = "onnx-community/OuteTTS-0.2-500M",
        language = "en",
        tokenizer_path = null,
        languages = [],
        verbose = false,
        device = null,
        dtype = null,
        additional_model_config = {},
        wavtokenizer_model_path = null,
        max_seq_length = 4096,
    } = {}) {
        this.model_path = model_path;
        this.language = language;
        this.tokenizer_path = tokenizer_path;
        this.languages = languages;
        this.verbose = verbose;
        this.device = device;
        this.dtype = dtype;
        this.additional_model_config = additional_model_config;
        this.wavtokenizer_model_path = wavtokenizer_model_path;
        this.max_seq_length = max_seq_length;
    }
}

export class ModelOutput {
    constructor({ audio, sr }) {
        /** @type {Tensor} */
        this.audio = audio;

        /** @type {number} */
        this.sr = sr;
    }

    // Adapted from https://www.npmjs.com/package/audiobuffer-to-wav
    to_wav() {
        if (this.audio.dims.length !== 2 || this.audio.dims[0] !== 1) {
            throw new Error(`Unsupported audio shape, expected [1, num_samples], got ${this.audio.dims}`);
        }
        /** @type {Float32Array} */
        const samples = this.audio.data;
        const sr = this.sr;

        let offset = 44;
        const buffer = new ArrayBuffer(offset + samples.length * 4);
        const view = new DataView(buffer);

        const writeString = (view, offset, string) => {
            for (let i = 0; i < string.length; ++i) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        /* RIFF identifier */
        writeString(view, 0, "RIFF");
        /* RIFF chunk length */
        view.setUint32(4, 36 + samples.length * 4, true);
        /* RIFF type */
        writeString(view, 8, "WAVE");
        /* format chunk identifier */
        writeString(view, 12, "fmt ");
        /* format chunk length */
        view.setUint32(16, 16, true);
        /* sample format (raw) */
        view.setUint16(20, 3, true);
        /* channel count */
        view.setUint16(22, 1, true);
        /* sample rate */
        view.setUint32(24, sr, true);
        /* byte rate (sample rate * block align) */
        view.setUint32(28, sr * 4, true);
        /* block align (channel count * bytes per sample) */
        view.setUint16(32, 4, true);
        /* bits per sample */
        view.setUint16(34, 32, true);
        /* data chunk identifier */
        writeString(view, 36, "data");
        /* data chunk length */
        view.setUint32(40, samples.length * 4, true);

        for (let i = 0; i < samples.length; ++i, offset += 4) {
            view.setFloat32(offset, samples[i], true);
        }

        return buffer;
    }

    async save(path) {
        const wave = this.to_wav();
        if (typeof window !== "undefined") {
            // Create blob
            const blob = new Blob([this.to_wav()], { type: "audio/wav" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = path;
            a.click();

            // Clean up
            URL.revokeObjectURL(url);
        } else {
            // Dynamically import fs
            const fs = await import("fs");
            fs.writeFileSync(path, Buffer.from(wave));
        }
    }
}

export class InterfaceHF {
    /**
     *
     */
    constructor({ config, audio_codec, prompt_processor, model }) {
        this.language = config.language;
        this.languages = config.languages;

        /** @param {HFModelConfig} */
        this.config = config;

        /** @type {AudioCodec} */
        this.audio_codec = audio_codec;

        /** @type {PromptProcessor} */
        this.prompt_processor = prompt_processor;

        /** @type {HFModel} */
        this.model = model;
    }

    /**
     * @param {HFModelConfig} config
     */
    static async load(config) {
        const prompt_processor = new PromptProcessor(
            await AutoTokenizer.from_pretrained(config.tokenizer_path ?? config.model_path),
            config.languages,
        );

        const model = new HFModel(
            await AutoModelForCausalLM.from_pretrained(config.model_path, {
                device: config.device,
                dtype: config.dtype,
            }),
        );

        const audio_codec = new AudioCodec(
            await PreTrainedModel.from_pretrained("onnx-community/WavTokenizer-large-speech-75token_decode", {
                // WebGPU is not currently supported, so we use the default device (WASM/CPU).
                // See https://github.com/microsoft/onnxruntime/issues/22994 for more information.
                // device: config.device,
                dtype: config.dtype,
            }),
        );

        return new InterfaceHF({
            config,
            audio_codec,
            prompt_processor,
            model,
        });
    }

    /**
     * Extract audio from the output tokens.
     * @param {bigint[]} tokens The output tokens.
     * @returns {Promise<Tensor>} The decoded audio.
     */
    async get_audio(tokens) {
        const output = this.prompt_processor.extract_audio_from_tokens(tokens);
        if (output.length === 0) {
            console.warn("No audio tokens found in the output");
            return null;
        }

        return this.audio_codec.decode(output);
    }

    print_default_speakers() {
        const total_speakers = Object.values(_DEFAULT_SPEAKERS).reduce(
            (total, speakers) => total + Object.keys(speakers).length,
            0,
        );
        console.log("\n=== ALL AVAILABLE SPEAKERS ===");
        console.log(`Total: ${total_speakers} speakers across ${Object.keys(_DEFAULT_SPEAKERS).length} languages`);
        console.log("-".repeat(50));

        for (const [language, speakers] of Object.entries(_DEFAULT_SPEAKERS)) {
            console.log(`\n${language.toUpperCase()} (${Object.keys(speakers).length} speakers):`);
            for (const speaker_name of Object.keys(speakers)) {
                console.log(`  - ${speaker_name}`);
            }
        }

        console.log("\nTo use a speaker: load_default_speaker(name)\n");
    }

    load_default_speaker(name) {
        const language = this.language.toLowerCase().trim();
        if (!(_DEFAULT_SPEAKERS[language] && _DEFAULT_SPEAKERS[language][name])) {
            throw new Error(`Speaker ${name} not found for language ${language}`);
        }
        return _DEFAULT_SPEAKERS[language][name];
    }

    /**
     *
     * @param {*} text
     * @param {*} speaker
     * @returns {Object}
     */
    prepare_prompt(text, speaker = null) {
        const prompt = this.prompt_processor.get_completion_prompt(text, this.language, speaker);

        return this.prompt_processor.tokenizer(prompt, {
            add_special_tokens: false,
        });
    }

    /**
     * Generate text using the model.
     * @param {Object} args The arguments for generation.
     * @param {string} args.text The input text to generate from.
     * @param {Object} [args.speaker=null] The speaker configuration.
     * @param {number} [args.temperature=0.1] The temperature for generation.
     * @param {number} [args.repetition_penalty=1.1] The repetition penalty for generation.
     * @param {number} [args.max_length=4096] The maximum length of the generated text.
     * @param {Object} [args.additional_gen_config={}] Additional generation configuration.
     * @returns {Promise<ModelOutput>} The generated model output.
     */
    async generate({
        text,
        speaker = null,
        temperature = 0.1,
        repetition_penalty = 1.1,
        max_length = 4096,
        additional_gen_config = {},
    }) {
        const inputs = this.prepare_prompt(text, speaker);

        if (this.config.verbose) {
            console.log(`Input tokens: ${inputs.input_ids.dims}`);
            console.log("Generating audio...");
        }

        const outputs = await this.model.generate({
            max_length,
            temperature,
            repetition_penalty,
            do_sample: true,
            ...additional_gen_config,
            ...inputs,
        });
        const new_tokens = outputs.slice(inputs.input_ids.dims[1]);

        const audio = await this.get_audio(new_tokens);

        if (this.config.verbose) {
            console.log("Audio generation completed");
        }

        return new ModelOutput({
            audio,
            sr: this.audio_codec.sr,
        });
    }
}
