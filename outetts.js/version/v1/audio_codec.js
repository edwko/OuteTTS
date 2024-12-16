import { Tensor } from "@huggingface/transformers";

export class AudioCodec {
    constructor(wavtokenizer) {
        this.wavtokenizer = wavtokenizer;
        this.sr = 24000;
    }

    /**
     * Decode a list of audio codes into a waveform.
     * @param {bigint[]} codes
     * @returns {Promise<Tensor>} The generated waveform.
     */
    async decode(codes) {
        codes = new Tensor("int64", codes, [1, codes.length]);
        const { waveform } = await this.wavtokenizer({ codes });
        return waveform;
    }
}
