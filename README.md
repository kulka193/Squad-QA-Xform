# Squad-QA-Xform
Question Answering with Transformer Model

## Summary: Question Answering with a Transformer Model

This notebook demonstrates the implementation of a Question Answering system using a custom Sequence-to-Sequence Transformer model. The project leverages the SQuAD v2 (Stanford Question Answering Dataset) to train and evaluate the model's ability to comprehend context and generate accurate answers to given questions.

### Key Features:

*   **Dataset Handling**: Efficient loading and preprocessing of the SQuAD v2 dataset, including handling of large datasets with custom `QADataset` and `ShuffleDataset` for improved training dynamics.
*   **Tokenization**: Utilizes `BertTokenizer` for robust text tokenization, managing special tokens and vocabulary.
*   **Custom Transformer Architecture**: Implements a `Seq2SeqTransformer` model from scratch using PyTorch, featuring an encoder-decoder structure with positional embeddings.
*   **Training Pipeline**: Includes a comprehensive training loop with `Adam` optimizer, `CrossEntropyLoss`, learning rate scheduling (`ReduceLROnPlateau`), and mixed precision training (`torch.cuda.amp`) for enhanced performance and efficiency.
*   **Inference Methods**: Explores both greedy decoding and beam search strategies for generating answers, showcasing different approaches to sequence generation.
*   **Model Persistence**: Demonstrates saving and loading of trained models, enabling reusability and further experimentation.

This notebook serves as a practical guide to building, training, and evaluating a transformer-based question answering system, highlighting key components and techniques involved in natural language processing tasks.
