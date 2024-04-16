# Sentiment Analysis

## Dataset 
- [IMDB sentiment analysis](https://huggingface.co/datasets/imdb)
- [Banking Intent classification](https://huggingface.co/datasets/banking77)
- [Twitter sentiment analysis](https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis)
<!-- - [Vietnamese sentiment analysis](https://github.com/congnghia0609/ntc-scv) -->

<!-- - [Penn Tree Bank] -->

## Models
- Transformer from scratch
- DistilBERT
- FLAN T5 (prompting)

## Results
- **IMDB sentiment analysis**:
| Model | F1 |
| ----------- | ----------- |
| Trasformer from scratch | 79.19% |
| DistilBERT | 93% |
| Zero-shot prompting | 90.75% |


## To-Do
- Banking Intent classification:
    - Fine-tune with custom pipeline
    - Fine-tune with hugging face pipline
- Twitter sentiment analysis:
    - Fine-tune with hugging face pipline
    - Instruction tuning (prompting) FLAN T5

