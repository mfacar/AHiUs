
from indexer.ElasticSearchHandler import ElasticSearchWriter
#
# try:
#     with open('./../word2vec.pickle', 'rb') as handle:
#         word2vec = pickle.load(handle)
# except FileNotFoundError:
#     word2vec = gensim.models.KeyedVectors.load_word2vec_format('./../resources/SBW-vectors-300-min5.bin',
#                                                                binary=True)
#     with open('./../word2vec.pickle', 'wb') as handle:
#         pickle.dump(word2vec, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_model_4():
    vector_dim = 300
    windows_size = 10

    docs = ElasticSearchWriter.search(index='idx_story', doc_type="story")


    text = "como algo esto es una prueba"

    # data = pd.read_csv('./../resources/feature_data.csv', sep='\t')
    # data['word_2_vec_feature'] = data['word_2_vec_feature'].apply(ast.literal_eval)
    # data["input"] = pad_sequences(data["word_2_vec_feature"], value=0, padding="post", maxlen=windows_size).tolist()
    # data = data.sample(frac=1).reset_index(drop=True)
    #
    # train_a = np.stack(data["input"], axis=0)
    #
    # data['target'] = data['target'].apply(ast.literal_eval)
    # data["target"] = pad_sequences(data["target"], value=0, padding="post", maxlen=windows_size).tolist()
    # data = data.sample(frac=1).reset_index(drop=True)
    # train_b = np.stack(data["target"], axis=0)
    #
    # vocab_size = len(word2vec.index2word)
    # embedding = word2vec.vectors
    #
    #
    # word2vec.cosine_similarities(train_a[0], train_a)
