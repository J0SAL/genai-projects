reference - https://youtu.be/bLDlwcl6hbA?si=wWG7ojoNvoP1IXhg

# Build a Gemma-style Small Language Model from Scratch

This project provides a step-by-step implementation of a small language model (SLM) from scratch using PyTorch, as detailed in the Jupyter Notebook [`Gemma_3_270_M_Small_Language_Model_Scratch_Final.ipynb`](3-gemma_slm/Gemma_3_270_M_Small_Language_Model_Scratch_Final.ipynb). The model architecture is inspired by Google's Gemma 3, with approximately 270 million parameters. The notebook guides you through data preparation, model definition, training, and inference on the TinyStories dataset.

## Features

-   **Model Architecture**: A custom transformer model inspired by Gemma 3, featuring:
    -   **Grouped-Query Attention (GQA)** for efficient inference.
    -   **RMSNorm** for layer normalization.
    -   **Rotary Positional Embeddings (RoPE)** for sequence position encoding.
    -   **Sliding Window Attention** combined with full attention layers.
-   **Dataset**: Uses the `roneneldan/TinyStories` dataset from Hugging Face, suitable for training small models.
-   **Training**: Implements a complete training pipeline, including:
    -   Mixed-precision training with `bfloat16` or `float16`.
    -   AdamW optimizer with weight decay.
    -   A learning rate scheduler with linear warmup and cosine decay.
    -   Gradient accumulation to simulate larger batch sizes.
-   **Inference**: A generation function to produce text from a given prompt.

## How to Run

1.  Launch Jupyter Notebook or an IDE with notebook support like VS Code.
2.  Open the [`Gemma_3_270_M_Small_Language_Model_Scratch_Final.ipynb`](3-gemma_slm/Gemma_3_270_M_Small_Language_Model_Scratch_Final.ipynb) file.
3.  Run the cells sequentially. The notebook will handle:
    -   Downloading and tokenizing the dataset.
    -   Defining the model architecture.
    -   Training the model and saving the best weights.
    -   Plotting the training and validation loss.
    -   Running inference to generate text.

## Model Configuration

The model uses the following configuration, defined in the notebook:

```python
GEMMA3_CONFIG_270M = {
    "vocab_size": 50257,
    "context_length": 32_768,
    "emb_dim": 640,
    "n_heads": 4,
    "n_layers": 18,
    "hidden_dim": 2048,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 1,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "sliding_window": 512,
      "layer_types": [
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention"
    ],
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 256,
```

## Key Concepts Explained

This project utilizes several modern techniques for building and training language models efficiently.

### Transformer Architecture
The entire model is based on the Transformer architecture, which has become the standard for language tasks. It processes the entire input sequence at once, using a mechanism called **self-attention** to weigh the importance of different words in the context of others. Our model is a stack of these Transformer blocks.

### Grouped-Query Attention (GQA)
This is an optimization of the standard multi-head attention mechanism. Instead of each "query" head having its own "key" and "value" heads, GQA allows multiple query heads to share a single key/value head. This significantly reduces the memory and computational requirements during inference, making the model faster and more efficient without a major loss in performance.

### RMSNorm (Root Mean Square Layer Normalization)
Instead of standard Layer Normalization, this model uses RMSNorm. It's a simpler and more computationally efficient normalization technique that helps stabilize the network's activations during training. It normalizes the activations based on their root mean square, which has been shown to be effective in large language models.

### Rotary Positional Embeddings (RoPE)
Transformers don't inherently understand the order of words. RoPE is a clever method for encoding the position of each token in the sequence. It "rotates" the query and key vectors based on their absolute position, allowing the self-attention mechanism to implicitly understand relative positions and distances between tokens.

### Sliding Window Attention
To handle very long sequences without running out of memory, some layers in our model use sliding window attention. Instead of calculating attention scores across the entire sequence, each token only attends to a fixed-size window of recent tokens (e.g., the last 512 tokens). This provides a good balance between computational efficiency and capturing local context.

### Mixed-Precision Training
This technique speeds up training and reduces GPU memory usage by performing many calculations using lower-precision floating-point numbers (`bfloat16` or `float16`) instead of the standard `float32`. A `GradScaler` is used to prevent numerical issues that can arise from using these lower-precision formats.

### Gradient Accumulation
This technique allows for training with a large effective batch size, even on a single GPU with limited memory. Instead of updating the model's weights after every batch, the gradients (error signals) are accumulated over several batches. The model's weights are only updated after the gradients from `gradient_accumulation_steps` batches have been summed up.

### Memory-Mapped Files (`memmap`)
The TinyStories dataset is very large. Instead of loading the entire tokenized dataset into RAM, we use `numpy.memmap`. This creates a mapping to the data stored on the hard disk, allowing the program to access small chunks of the data as if it were in memory, which is far more memory-efficient.