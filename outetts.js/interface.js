import { InterfaceHF as _InterfaceHF_v1, HFModelConfig as HFModelConfig_v1 } from "./version/v1/interface.js";

const MODEL_CONFIGS = Object.freeze({
    // TODO: Add support for 0.1 model
    // 0.1: {
    //     tokenizer: "onnx-community/OuteTTS-0.1-350M",
    //     sizes: ["350M"],
    //     links: ["https://huggingface.co/onnx-community/OuteTTS-0.1-350M"],
    //     languages: ["en"],
    //     hf_interface: _InterfaceHF_v1,
    //     max_seq_length: 4096,
    // },
    0.2: {
        tokenizer: "onnx-community/OuteTTS-0.2-500M",
        sizes: ["500M"],
        links: ["https://huggingface.co/onnx-community/OuteTTS-0.2-500M"],
        languages: ["en", "ja", "ko", "zh"],
        hf_interface: _InterfaceHF_v1,
        max_seq_length: 4096,
    },
});

function display_available_models() {
    console.log("\n=== Available OuteTTS Models ===\n");
    const separator = "-".repeat(50);
    for (const [version, details] of Object.entries(MODEL_CONFIGS)) {
        console.log(separator);
        console.log(`Version: ${version}`);
        console.log(`Supported Languages: ${details.languages.join(", ")}`);
        console.log(`Model Sizes: ${details.sizes.join(", ")}`);
        console.log("Available Formats: HF");
        console.log(`Tokenizer: ${details.tokenizer}`);
        console.log(`Links: ${details.links.join(", ")}`);
        console.log(separator + "\n");
    }
}

/**
 * Retrieve the configuration for a given model version.
 * @param {keyof MODEL_CONFIGS} version Version identifier for the model.
 */
function get_model_config(version) {
    if (!(version in MODEL_CONFIGS)) {
        throw new Error(
            `Unsupported model version '${version}'. Supported versions are: ${Object.keys(MODEL_CONFIGS)}`,
        );
    }
    return MODEL_CONFIGS[version];
}

function check_max_length(max_seq_length, model_max_seq_length) {
    if (!max_seq_length) {
        throw new Error("max_seq_length must be specified.");
    }
    if (max_seq_length > model_max_seq_length) {
        throw new Error(
            `Requested max_seq_length (${max_seq_length}) exceeds the maximum supported length (${model_max_seq_length}).`,
        );
    }
}

/**
 * @typedef {Object} InterfaceConstructorArgs
 * @property {string} model_version Version identifier for the model to be loaded.
 * @property {HFModelConfig_v1} cfg Configuration object containing parameters.
 * @property {string} cfg.tokenizer_path Path to the tokenizer.
 * @property {string} cfg.language Language to be used.
 * @property {number} cfg.max_seq_length Maximum sequence length.
 */

/**
 * Creates and returns a Hugging Face model interface for OuteTTS.
 *
 * @param {InterfaceConstructorArgs} inputs
 * @returns {Promise<_InterfaceHF_v1>} An instance of the interface based on the specified version.
 * @throws {Error} If the specified language is not supported by the model version.
 */
export async function InterfaceHF({ model_version, cfg }) {
    const config = get_model_config(model_version);
    cfg.tokenizer_path = cfg.tokenizer_path || config.tokenizer;
    const languages = config.languages;
    if (!languages.includes(cfg.language)) {
        throw new Error(
            `Language '${cfg.language}' is not supported by model version '${model_version}'. Supported languages are: ${languages}`,
        );
    }
    cfg.languages = languages;

    const interface_class = config.hf_interface;

    check_max_length(cfg.max_seq_length, config.max_seq_length);

    return await interface_class.load(cfg);
}

export { HFModelConfig_v1 };
