import { number_to_words } from './utils/number_to_words.js';

export class PromptProcessor {
    /**
     *
     * @param {import('@huggingface/transformers').PreTrainedTokenizer} tokenizer
     * @param {string[]} languages
     */
    constructor(tokenizer, languages) {
        this.tokenizer = tokenizer;
        this.bos = "<|im_start|>";
        this.eos = "<|im_end|>";
        this.special_tokens = {
            audio_code: "<|{}|>",
            text_start: "<|text_start|>",
            text_end: "<|text_end|>",
            audio_start: "<|audio_start|>",
            audio_end: "<|audio_end|>",
            time: "<|t_{:.2f}|>",
            code_start: "<|code_start|>",
            code_end: "<|code_end|>",
            text_sep: "<|text_sep|>",
        };
        this.text_prompt = "{bos}\n{text_start}{words}{text_end}\n{audio_start}\n";
        this.map_audio_tokens = this.get_audio_token_map();

        this.languages = languages;
    }

    get_audio_token_map() {
        const map = new Map();
        for (let i = 0; i < 4100; ++i) {
            const token = this.tokenizer.encode(this.special_tokens.audio_code.replace("{}", i), {
                add_special_tokens: false,
            })[0];
            map.set(BigInt(token), i);
        }
        return map;
    }

    process_text(text, language) {
        if (!this.languages.includes(language)) {
            throw new Error(`Language ${language} not supported, supported languages are ${this.languages}`);
        }
        if (language !== "en") {
            throw new Error("Non-English languages are not supported yet.");
        }

        text = text
            .toLowerCase()
            .replace(/\d+(\.\d+)?/g, match => number_to_words(Number(match)))
            .replace(/[-_/,\.\\]/g, " ")
            .replace(/[^a-z\s]/g, "")
            .replace(/\s+/g, " ")
            .trim();
        return text.split(" ");
    }

    create_audio_prompt(words) {
        return words
            .map((i) => {
                const word = i.word;
                const duration = this.special_tokens.time.replace("{:.2f}", i.duration.toFixed(2));
                const tokens = i.codes.map((c) => this.special_tokens.audio_code.replace("{}", c)).join("");
                return `${word}${duration}${this.special_tokens.code_start}${tokens}${this.special_tokens.code_end}`;
            })
            .join("\n");
    }

    get_completion_prompt(text, language, speaker = null) {
        let words = this.process_text(text, language);
        if (speaker !== null) {
            if (speaker.language !== language) {
                console.warn(`Speaker language ${speaker.language} does not match text language ${language}`);
            }
            words = this.process_text(speaker.text, speaker.language).concat(words);
        }

        words = words.map((word) => word.trim()).join(this.special_tokens.text_sep);

        let prompt = this.text_prompt
            .replace("{bos}", this.bos)
            .replace("{text_start}", this.special_tokens.text_start)
            .replace("{words}", words)
            .replace("{text_end}", this.special_tokens.text_end)
            .replace("{audio_start}", this.special_tokens.audio_start);

        if (speaker !== null) {
            prompt += this.create_audio_prompt(speaker.words);
        }

        return prompt;
    }

    /**
     *
     * @param {bigint[]} tokens
     * @returns {bigint[]}
     */
    extract_audio_from_tokens(tokens) {
        const result = [];
        for (const token of tokens) {
            const x = this.map_audio_tokens.get(token);
            if (x) result.push(x);
        }
        return result;
    }
}
