from glove import Corpus, Glove

def read_bytecode_corpus(corpus):
    instruction_list = []

    for instruction in corpus.split('\n'):
        instruction_list.append(instruction.replace(',', ' ').split())

    return instruction_list

def glove_model(bytecode_corpus, filename):
    word_list = read_bytecode_corpus(bytecode_corpus)

    corpus = Corpus()
    corpus.fit(word_list, window=10)

    glove = Glove(no_components=vectorsize, learning_rate=0.01)
    glove.fit(corpus.matrix, epochs=50, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)

    ret_str = ''

    for key in glove.dictionary:
        vector = \
            str(glove.word_vectors[glove.dictionary[key]]).replace('\n'
                , ' ').replace(',', '')
        ret_str += key + ' ' + vector.strip('[]') + '\n'
    return ret_str

with open('bytecode_corpus.txt', 'r') as bytecode_corpus_file:
    bytecode_corpus = bytecode_corpus_file.read()

with open('bytecode_embeddings_glove.txt', 'w') as text_file:
    text_file.write(glove_model(bytecode_corpus, 'glove.model'))
