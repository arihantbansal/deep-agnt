import numpy as np
import torch
import torch.nn.functional as F

def embed(sentences, tokenizer, model):
    """Return normalized sentence embeddings, shape (n, d), as a NumPy array."""
    
    tokenized = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**tokenized)

    last_hidden_state = outputs.last_hidden_state
    
    attention_mask = tokenized["attention_mask"]  # (batch_size, seq_len)
    # 1. Expand attention mask to match hidden_state dimensions
    mask_expanded = (
        attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    )

    # 2. Sum valid token embeddings and divide by actual token count (ignoring padding)
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    sentence_embeddings = sum_embeddings / sum_mask  # (batch_size, hidden_size)

    # 3. L2 normalize the pooled sentence embeddings
    normalized_embeddings = F.normalize(sentence_embeddings, p=2, dim=-1)

    return np.asarray(normalized_embeddings)
