#!/bin/sh

# python3 simple_model.py random static 0&
# python3 simple_model.py random dynamic 0&
# python3 simple_model.py fasttext static 4&
# python3 simple_model.py fasttext dynamic 4&
# python3 simple_model.py glove static 5&
# python3 simple_model.py glove dynamic 5&
# python3 simple_model.py word2vec static 1&
# python3 simple_model.py word2vec dynamic 4&
# python3 CNN_model.py random static 0 0&
# python3 CNN_model.py random dynamic 1 0&
# python3 CNN_model.py fasttext static 4 0&
# python3 CNN_model.py fasttext dynamic 5 0&
# python3 CNN_model.py glove static 1 0&
# python3 CNN_model.py glove dynamic 2 0&
# python3 CNN_model.py word2vec static 0 0&
# python3 CNN_model.py word2vec dynamic 4 0&
# python3 CNN_model.py random static 5 0.5&
# python3 CNN_model.py random dynamic 4 0.5&
# python3 CNN_model.py fasttext static 5 0.5&
# python3 CNN_model.py fasttext dynamic 0 0.5&
# python3 CNN_model.py glove static 2 0.5&
# python3 CNN_model.py glove dynamic 3 0.5&
# python3 CNN_model.py word2vec static 0 0.5&
# python3 CNN_model.py word2vec dynamic 2 0.5&
# python3 CNN_model.py glove dynamic 0 0.1&
# python3 CNN_model.py glove dynamic 2 0.2&
# python3 CNN_model.py glove dynamic 0 0.4&
# python3 CNN_model.py glove dynamic 3 0.3&

python3 CNN_model.py glove dynamic 0 0 32&
python3 CNN_model.py glove dynamic 2 0 64&
python3 CNN_model.py glove dynamic 2 0 128&
python3 CNN_model.py glove dynamic 3 0 16&