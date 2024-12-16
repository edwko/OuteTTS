# Training Instructions

The model can be trained similarly to other transformer-based models. An example for preparing datasets is included in `examples/v1/data_creation.py`. After generating the dataset, you can begin training using your preferred library. Below are some suggested libraries for tasks like supervised fine-tuning (SFT):

- [Hugging Face's SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [TorchTune](https://github.com/pytorch/torchtune)

Refer to the respective documentation for detailed setup and instructions.