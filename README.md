# Qwen2.5-3B-Instruct: LoRA Fine-Tuning

This project explores parameter-efficient fine-tuning of Qwen2.5-3B-Instruct using Low-Rank Adaptation (LoRA) and a custom-created dataset for a Duolingo-related task.

The goal was to adapt a pretrained large language model using PEFT and monitor its training behavior through recorded loss values.

## Project Overview

Full fine-tuning updates all model parameters, which can require substantial computational resources. LoRA provides a parameter-efficient alternative by introducing trainable low-rank matrices into selected layers of a pretrained model.

This project uses LoRA to fine-tune Qwen2.5-3B-Instruct while keeping the original model weights frozen.

### Key Objectives

- Fine-tune a pretrained 3-billion-parameter language model.
- Apply LoRA using the Hugging Face PEFT library.
- Prepare a custom dataset with AI assistance.
- Monitor training loss and analyze optimization behavior.
- Explore parameter-efficient adaptation of instruction-tuned language models.

## Model and Dataset

**Base Model:** [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

**Dataset:** Custom-created training data prepared with assistance from Claude.

The dataset was created for the project's Duolingo-related task and used to adapt the pretrained model through supervised fine-tuning.

## LoRA Configuration

The model was fine-tuned using the following PEFT configuration:

```python
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.5,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    task_type="CAUSAL_LM"
)
```

### Configuration Details

| Parameter | Value | Description |
|---|---|---|
| `r` | 16 | Rank of the LoRA update matrices |
| `lora_alpha` | 32 | Scaling parameter for LoRA updates |
| `lora_dropout` | 0.5 | Dropout applied within the LoRA layers |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Attention projection layers adapted by LoRA |
| `task_type` | `CAUSAL_LM` | Configures PEFT for causal language modeling |

The LoRA adapters target the query, key, value, and output projections in the model's attention layers.

The rank of 16 determines the dimensionality of the low-rank updates, while `lora_alpha=32` controls their scaling.

The configured dropout of 0.5 introduces regularization during training.

## Training Results

The following table shows the recorded training loss at different training steps.

| Training Step | Training Loss |
|---:|---:|
| 10 | 3.945908 |
| 20 | 2.035421 |
| 30 | 1.708151 |
| 40 | 1.453950 |
| 50 | 1.328866 |
| 60 | 1.260986 |
| 70 | 1.213928 |
| 80 | 1.149541 |
| 90 | 1.088966 |
| 100 | 1.098806 |
| 110 | 1.128118 |

### Loss Analysis

The training loss decreased substantially during the observed training process.

- **Initial recorded loss:** 3.9459 at step 10.
- **Lowest recorded loss:** 1.0890 at step 90.
- **Loss reduction:** Approximately 72.4% between steps 10 and 90.
- **Later training behavior:** A slight increase in loss occurred at steps 100 and 110.

The decreasing loss suggests that the model was progressively fitting the training data. The small increase in later steps could reflect training fluctuations or a plateau.

These values represent training loss only. Validation metrics and task-specific tests would be needed to assess generalization and determine whether the fine-tuned model improves on the original model.

## Technologies Used

- Python
- Qwen2.5-3B-Instruct
- PyTorch
- Hugging Face Transformers
- Hugging Face PEFT
- LoRA
- Supervised fine-tuning

## Workflow

1. Load the pretrained Qwen2.5-3B-Instruct model.
2. Prepare a custom dataset with AI assistance.
3. Configure LoRA adapters using PEFT.
4. Apply adapters to the selected attention projection layers.
5. Fine-tune the model on the prepared dataset.
6. Monitor training loss at regular intervals.
7. Analyze the resulting training curve.

## Key Takeaways

This project provided practical experience with parameter-efficient fine-tuning of a pretrained instruction-following language model.

It demonstrates the use of LoRA to adapt selected attention layers, the preparation of custom training data, and the monitoring of optimization through training loss.

The lowest recorded training loss was approximately 1.089 at step 90. Further evaluation would be necessary to determine how well the model performs on unseen examples and how its outputs compare with those of the original model.

## Future Improvements

- Evaluate the fine-tuned model on a separate validation dataset.
- Compare the original and fine-tuned models using identical prompts.
- Test generated outputs against task-specific quality criteria.
- Investigate the later increase in training loss.
- Experiment with different LoRA ranks and dropout values.
- Document the training hardware, dataset size, and hyperparameters.
- Publish example inputs and outputs to demonstrate the effect of fine-tuning.
