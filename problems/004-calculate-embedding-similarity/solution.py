import numpy as np


def cosine_similarity(query, documents):
    q = np.asarray(query, dtype=float)
    d = np.asarray(documents, dtype=float)

    dot = d @ q

    mag_q = np.sqrt(np.sum(q**2))
    mag_d = np.sqrt(np.sum(d**2, axis=1))

    if mag_q == 0:
        return np.zeros(len(d))
    
    denom = mag_q * mag_d
    return np.where(denom == 0, 0.0, dot / denom)

