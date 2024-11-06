import gensim
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim
import pandas as pd
import numpy as np


def compute_coherence_values(
    corpus,
    texts,
    id2word,
    start,
    limit,
    step,
) -> tuple[list,list]:
    coherence_values = []
    perplexity_values = []
    model_list = []
    nr_of_topics = range(start, limit, step)
    for num_topics in nr_of_topics:
        model = gensim.models.ldamodel.LdaModel(
            corpus=corpus, 
            id2word = id2word, 
            num_topics=num_topics,
            random_state=100,
            # update_every=1,
            # chunksize=100,
            # passes=10,
            # alpha='auto',
            # per_word_topics=True,
        )
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=id2word, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        perplexity_values.append(model.log_perplexity(corpus))

    return model_list, coherence_values, perplexity_values, list(nr_of_topics)


def get_best_model(index, models):
    return models[index]


def create_coherence_plot(nr_of_topics, coherence_values, path):
    plt.plot(nr_of_topics, coherence_values)
    plt.xlabel('Num Topics')
    plt.ylabel('Coherence Score')
    plt.legend(('coherence_values'), loc='best')
    plt.savefig(path)
    plt.show()


def save_coherence_values(nr_of_topics, coherence_values, perplexity_values, path):
    df = pd.DataFrame(
        {
            "nr_topics":nr_of_topics,
            "coherence_values":coherence_values,
            "perplexity_values":perplexity_values
        }
    )
    df.to_csv(path_or_buf=path, index=False)
    return df


def create_intertopic_distance_map(lda_model, corpus, id2word, path):
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, path)
    return vis


def save_lda_topics(lda_model, path, **kwargs):
    topics= lda_model.print_topics(**kwargs)
    rows = []

    for topic, words_weights in topics:
        pairs = words_weights.split(" + ")
        for pair in pairs:
            weight, word = pair.split("*")
            word = word.replace('"', '')
            rows.append((topic, word, float(weight)))

    df = pd.DataFrame(rows, columns=["topic", "word", "weight"])
    df.to_csv(path_or_buf=path, index=False)
    return df


