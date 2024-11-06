from nlp.lda_model.lda_analysis import *
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim

def main(app_name, series, nr_of_topics, nr_of_words_per_topic, start, limit, step):
    series = series.copy()
    data = series.tolist()
    id2word = corpora.Dictionary(data)
    texts = data
    corpus = [id2word.doc2bow(text) for text in texts]
    
    
    model_list, coherence_values, perplexity_values, topics = compute_coherence_values(
        corpus=corpus,
        texts=texts,
        id2word=id2word,
        start=start,
        limit=limit,
        step=step
    )
    save_coherence_values(
        nr_of_topics=topics,
        coherence_values=coherence_values,
        perplexity_values=perplexity_values,
        path= f"nlp/results_lda/topic_vs_coherence_data/{app_name}.csv"
    )
    create_coherence_plot(
        nr_of_topics=topics,
        coherence_values=coherence_values,
        path= f"nlp/results_lda/topics_vs_coherence_plot/{app_name}.png"
    )
    
    best_nr_of_topics = input("Pick Best Nr of Topics: ")
    best_nr_of_topics = int(best_nr_of_topics)
    best_model_idx = topics.index(best_nr_of_topics)
    best_model = model_list[best_model_idx]
    
    word_weight_path = f"nlp/results_lda/topic_word_weights_data/{app_name}.csv"
    _ = save_lda_topics(
        lda_model=best_model,
        path=word_weight_path,
        num_topics=best_nr_of_topics,
        num_words=nr_of_words_per_topic
    )
    
    
    vis = create_intertopic_distance_map(
        lda_model=best_model,
        corpus=corpus,
        id2word=id2word,
        path=f"nlp/results_lda/intertopic_distance_map/{app_name}.html"
    )
    return vis



