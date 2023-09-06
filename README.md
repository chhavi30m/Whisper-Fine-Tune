# Whisper-Fine-Tune
[OpenAI's Whisper model](https://openai.com/research/whisper) is a state-of-the-art automatic speech recognition (ASR) model. It is trained on an end-to-end, encoder-decoder transformer, which breaks audio into 30-second chunks, convert it to a log-mel spectrogram and encode it. A decoder predicts a text caption, which can be language identification and speech transcription.<br>
I fine-tuned the model on Google's [Fleurs Hindi dataset](https://huggingface.co/datasets/google/fleurs) available on HuggingFace.<br>
The steps can be broadly written as follows:<br>
1. **Analyzing the dataset**: Finding the splits in the dataset and using a streaming dataset of the Hindi split
2. **Sampling**: Matching the sampling rate of the dataset and model to 16000
3. **Seq2Seq**: to make the training pipeline
4. **Data Collation**: Making pytorch tensors
5. **WER**: Evaluation on Word Error Rate
6. **WhisperForConditionalGeneration**: To load checkpoints for Whisper-medium
### References
1. [https://huggingface.co/docs/transformers/model_doc/whisper](https://huggingface.co/docs/transformers/model_doc/whisper)
2. [https://openai.com/research/whisper](https://openai.com/research/whisper)
3. [https://github.com/huggingface/transformers/blob/main/src/transformers/training_args_seq2seq.py](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args_seq2seq.py)
4. [https://github.com/vasistalodagala/whisper-finetune/blob/master/train/fine-tune_on_hf_dataset.py](https://github.com/vasistalodagala/whisper-finetune/blob/master/train/fine-tune_on_hf_dataset.py)
