import torch
import numpy as np


def get_detm_topics(beta, time, num_words, vocab, num_topics):
    with torch.no_grad(): 
        topics_words = []
        for k in range(num_topics):
            gamma = beta[k, time, :]
            top_words = list(gamma.cpu().numpy().argsort()[-num_words:][::-1])
            topics_words.append([vocab[a] for a in top_words])
    return topics_words


def topic_diversity(topics):
    all_words = set(np.concatenate(topics))
    return len(all_words) / (len(topics[0]) * len(topics))