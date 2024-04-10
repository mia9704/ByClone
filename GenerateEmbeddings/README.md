# Generate Your Own Embeddings
The embedding files used in ByClone that are already trained are available at ByClone/bytecode_embeddings_word2vec.txt, ByClone/bytecode_embeddings_fasttext.txt, ByClone/bytecode_embeddings_glove.txt, and ByClone/bytecode_embeddings_instruction2vec.txt. 
However if you would like to generate them yourself, you can run the scripts provided in this directory for Word2Vec, FastText, and GloVe. You can create your own Instruction2Vec embedding using the following link: https://github.com/firmcode/instruction2vec

## Requirements

-Python 3.8.16

-glove-python-binary 0.2.0

-gensim 4.3.2

## Usage
```python
pip install -r requirements.txt
python fasttext_embedding.py
python word2vec_embedding.py
python glove_embedding.py
```
