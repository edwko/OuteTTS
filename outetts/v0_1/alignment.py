import torchaudio
import torch
import torchaudio.functional as F
import inflect
import re

class CTCForcedAlignment:
    # Implimentation of CTC forced alignment
    # https://pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = torchaudio.pipelines.MMS_FA
        self.sample_rate = bundle.sample_rate
        self.model = bundle.get_model(with_star=False).to(self.device)
        self.LABELS = bundle.get_labels(star=None)
        self.DICTIONARY = bundle.get_dict(star=None)
        self.lec = inflect.engine()
        
    def process_text(self, text: str):
        text = re.sub(r'\d+(\.\d+)?', lambda x: self.lec.number_to_words(x.group()), text.lower())
        text = re.sub(r'[^a-z\s]', '', text)
        return text.split()
    
    def _unflatten(self, list_, lengths):
        assert len(list_) == sum(lengths)
        i = 0
        ret = []
        for l in lengths:
            ret.append(list_[i : i + l])
            i += l
        return ret

    def get_word(self, waveform, spans, num_frames, transcript):
        ratio = waveform.size(1) / num_frames
        x0 = int(ratio * spans[0].start)
        x1 = int(ratio * spans[-1].end)
        return {"x0": x0, "x1": x1, "word": transcript}
    
    def _extract_world_level(self, aligned_tokens, alignment_scores, transcript):
        token_spans = F.merge_tokens(aligned_tokens, alignment_scores)
        word_spans = self._unflatten(token_spans, [len(word) for word in transcript])
        return word_spans

    def _align(self, emission, tokens):
        targets = torch.tensor([tokens], dtype=torch.int32, device=torch.device("cpu"))
        alignments, scores = F.forced_align(emission.cpu(), targets, blank=0)
        alignments, scores = alignments[0], scores[0] 
        scores = scores.exp()  
        return alignments, scores

    def align(self, audio, transcript):
        waveform, sr = torchaudio.load(audio)
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
        transcript = self.process_text(transcript)

        with torch.inference_mode():
            emission, _ = self.model(waveform.to(self.device))

        tokenized_transcript = [self.DICTIONARY[c] for word in transcript for c in word]
        alignments, scores = self._align(emission, tokenized_transcript)
        word_spans = self._extract_world_level(alignments, scores, transcript)
        num_frames = emission.size(1)

        outputs = [
            self.get_word(waveform, word_spans[i], num_frames, transcript[i]) 
            for i in range(len(word_spans))
        ]

        outputs[0]["x0"] = 0

        for i in range(len(outputs)):
            output = outputs[i]
            x0 = output["x0"]

            if i == len(outputs) - 1:
                x1 = output["x1"]
            else:
                x1 = outputs[i + 1]["x0"]

            outputs[i]["audio"] = waveform[:, x0:x1]

        return outputs
    
    def free(self):
        del self.model