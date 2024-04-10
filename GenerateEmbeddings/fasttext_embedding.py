from gensim.models import FastText

def read_bytecode_corpus(corpus):
	instruction_list = []

	for instruction in corpus.split('\n'):
		instruction_list.append(instruction.replace(',',' ').split())

	return instruction_list

def fasttext_model(asmcode_corpus, filename):

	word_list = read_bytecode_corpus(asmcode_corpus)

	model = FastText(word_list, vector_size=300, window=128, min_count=1, workers=4, epochs = 10)

	model.save(filename)
	ret_str = ""

	for key in model.wv.key_to_index.keys():
		vector = str(model.wv[key]).replace("\n", " ").replace(",", "")
		ret_str += key + " " + vector.strip("[]") +"\n"
	return ret_str

with open("bytecode_corpus.txt", "r") as bytecode_corpus_file:
    bytecode_corpus = bytecode_corpus_file.read()

with open("bytecode_embeddings_fasttext.txt", "w") as text_file:
    text_file.write(fasttext_model(bytecode_corpus, "fasttext.model"))
