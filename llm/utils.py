import torch
import numpy as np

def get_tokens_ids(
    tokenizer,
    inputs:list,
    prefixes = [],
    suffixes = [],
    token_id = -1,
    check_decode:bool=False # TODO
):
    tokens_ids_dict = {
        input: list(set(
            [tokenizer.encode(input)[token_id]]+[ # keep first token
                tokenizer.encode(pre+input)[token_id] # !!! quite sketchy
                for pre in prefixes
            ]+[
                tokenizer.encode(input+suf)[token_id]
                for suf in suffixes
            ]
        ))
        for input in inputs
    }

    if check_decode:
        for k, values in tokens_ids_dict.items():
            tokens_ids_dict[k] = [
                v for v in values
                if k in tokenizer.decode(v)
            ]
    
    return tokens_ids_dict

def get_tokens_prob(
    logits,
    token_ids:list,
    normalize:bool=True,
):
    soft_m = torch.nn.functional.softmax(logits).to('cpu')[0]
    probs = {
        k: np.sum([soft_m[id] for id in ids])
        for k, ids in token_ids.items()
    }
    
    if normalize:
        tot = np.sum(list(probs.values()))
        for k in probs.keys():
            probs[k] = probs[k]/tot
    
    return probs