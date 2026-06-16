# Model report

- **Generated**: `2026-06-14 08:53:59`
- **Weights file**: `C:\Users\andre\Desktop\ser_andreaingoglia_unimi\wav2vec2-lg-xlsr-en-speech-emotion-recognition\pytorch_model.bin`
- **Repo root**: `C:\Users\andre\Desktop\ser_andreaingoglia_unimi\wav2vec2-lg-xlsr-en-speech-emotion-recognition`

## Weight statistics
- **Total parameters (by tensor elements)**: `316,496,520`
- **Dtypes**: `torch.float32`

### Largest tensors (top 20)
- `wav2vec2.encoder.pos_conv_embed.conv.weight_v` — shape `(1024, 64, 128)` — params `8,388,608`
- `wav2vec2.encoder.layers.9.feed_forward.output_dense.weight` — shape `(1024, 4096)` — params `4,194,304`
- `wav2vec2.encoder.layers.9.feed_forward.intermediate_dense.weight` — shape `(4096, 1024)` — params `4,194,304`
- `wav2vec2.encoder.layers.8.feed_forward.output_dense.weight` — shape `(1024, 4096)` — params `4,194,304`
- `wav2vec2.encoder.layers.8.feed_forward.intermediate_dense.weight` — shape `(4096, 1024)` — params `4,194,304`
- `wav2vec2.encoder.layers.7.feed_forward.output_dense.weight` — shape `(1024, 4096)` — params `4,194,304`
- `wav2vec2.encoder.layers.7.feed_forward.intermediate_dense.weight` — shape `(4096, 1024)` — params `4,194,304`
- `wav2vec2.encoder.layers.6.feed_forward.output_dense.weight` — shape `(1024, 4096)` — params `4,194,304`
- `wav2vec2.encoder.layers.6.feed_forward.intermediate_dense.weight` — shape `(4096, 1024)` — params `4,194,304`
- `wav2vec2.encoder.layers.5.feed_forward.output_dense.weight` — shape `(1024, 4096)` — params `4,194,304`
- `wav2vec2.encoder.layers.5.feed_forward.intermediate_dense.weight` — shape `(4096, 1024)` — params `4,194,304`
- `wav2vec2.encoder.layers.4.feed_forward.output_dense.weight` — shape `(1024, 4096)` — params `4,194,304`
- `wav2vec2.encoder.layers.4.feed_forward.intermediate_dense.weight` — shape `(4096, 1024)` — params `4,194,304`
- `wav2vec2.encoder.layers.3.feed_forward.output_dense.weight` — shape `(1024, 4096)` — params `4,194,304`
- `wav2vec2.encoder.layers.3.feed_forward.intermediate_dense.weight` — shape `(4096, 1024)` — params `4,194,304`
- `wav2vec2.encoder.layers.23.feed_forward.output_dense.weight` — shape `(1024, 4096)` — params `4,194,304`
- `wav2vec2.encoder.layers.23.feed_forward.intermediate_dense.weight` — shape `(4096, 1024)` — params `4,194,304`
- `wav2vec2.encoder.layers.22.feed_forward.output_dense.weight` — shape `(1024, 4096)` — params `4,194,304`
- `wav2vec2.encoder.layers.22.feed_forward.intermediate_dense.weight` — shape `(4096, 1024)` — params `4,194,304`
- `wav2vec2.encoder.layers.21.feed_forward.output_dense.weight` — shape `(1024, 4096)` — params `4,194,304`

## Configuration (from config.json)
- **architectures**: `['Wav2Vec2ForSequenceClassification']`
- **model_type**: `wav2vec2`
- **num_hidden_layers**: `24`
- **hidden_size**: `1024`
- **intermediate_size**: `4096`
- **num_attention_heads**: `16`
- **num_feat_extract_layers**: `7`
- **pooling_mode**: `mean`
- **num_labels**: `8`
- **final_dropout**: `0.0`

### Labels (id2label)
- `0`: `angry`
- `1`: `calm`
- `2`: `disgust`
- `3`: `fearful`
- `4`: `happy`
- `5`: `neutral`
- `6`: `sad`
- `7`: `surprised`

## Preprocessor (from preprocessor_config.json)
- **sampling_rate**: `16000`
- **do_normalize**: `True`
- **return_attention_mask**: `True`
- **feature_size**: `1`
- **padding_value**: `0.0`

## Training arguments (from training_args.bin)
```
[Could not load training_args.bin: UnpicklingError('Weights only load failed. This file can still be loaded, to do so you have two options, \x1b[1mdo those steps only if you trust the source of the checkpoint\x1b[0m. \n\t(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.\n\t(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.\n\tWeightsUnpickler error: Unsupported global: GLOBAL transformers.training_args.TrainingArguments was not an allowed global by default. Please use `torch.serialization.add_safe_globals([transformers.training_args.TrainingArguments])` or the `torch.serialization.safe_globals([transformers.training_args.TrainingArguments])` context manager to allowlist this global if you trust this class/function.\n\nCheck the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.')]
```

## Classification head-related tensors (from weights)
These keys strongly indicate the classifier/projector layers present in the checkpoint.
- `classifier.dense.bias` — shape `(1024,)`
- `classifier.dense.weight` — shape `(1024, 1024)`
- `classifier.output.bias` — shape `(8,)`
- `classifier.output.weight` — shape `(8, 1024)`
