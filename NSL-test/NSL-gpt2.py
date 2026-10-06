import numpy as np
import torch
import time
import math
torch.set_printoptions(8)
# KV_CACHE[layer][head] = {"k":[tensor(1, d_head),……], "V":[tensor(1,d_head), ……]}
KV_CACHE = []
USE_CACHE = True

def init_kv_cache(n_layer: int, n_head: int) -> list:
    return [[{"K":[],"V":[]} for _ in range(n_head)] for _ in range(n_layer)]

def reset_kv_cache():
    global KV_CACHE
    KV_CACHE = []

def cache_len() -> int:
    if not KV_CACHE or not KV_CACHE[0]:
        return 0
    return len(KV_CACHE[0][0]["K"])

def use_cache() -> bool:
    return USE_CACHE and cache_len() > 0

def gelu(x):
    """
        Task: Use the torch API to implement the approximate calculation formula of the `GELU`
        activation function. The formula is as follows (you need to paste it into the latex
        online conversion website)
        Website: https://www.latexlive.com/
        Formula: \frac{1}{2} x\left[1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right]
        
        Input: Tensor
        Output: Tensor
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3 )))


def softmax(x):
    """
        Task: Use torch API to implement `softmax` function, search the specific formula by yourself
        Input: Tensor
        Output: Tensor
    """
    x = x - x.max(dim = -1, keepdim = True).values
    e = torch.exp(x)
    return e / e.sum(dim = -1, keepdim = True)


def layer_norm(x, g_b, eps:float = 1e-5):
    """
        Task: Use torch API to implement `layernorm` function, search `layernorm` by yourself
        Input: 
            x: Tensor
            g_b: dictionary that load from gpt2 weight. g-gamma and b-bias are the keys
        Output: Tensor
    """
    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])

    mu = x.mean(dim = -1, keepdim = True)
    var = ((x - mu) ** 2).mean(dim = -1, keepdim = True)
    return (x - mu) / torch.sqrt(var + eps) * g + b


def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        Task: implement linear layer 
        Input: 
            x: Tensor
            w_b: dictionary that load from gpt2 weight. w-weight and b-bias are the keys
        Output: Tensor
    """
    w, b = w_b['w'], w_b['b']
    w, b = torch.as_tensor(w_b['w']), torch.as_tensor(w_b['b'])
    return x @ w + b
    

def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: use `gelu` `linear` to implement ffn
        Notes: x --linear--> --gelu--> --linear--> output
        Input: 
            x: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']
    x = linear(x, w_b1)
    x = gelu(x)
    x = linear(x, w_b2)
    return x

def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
        Task: use torch API to implement attention computation according to formula(1) of the following paper
              where d_k account for the last dimension of `k`
        Paper: https://arxiv.org/abs/1706.03762
        Input: 
            q: Tensor
            k: Tensor
            v: Tensor
            mask: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    d_k = k.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    scores = scores + mask
    weights = softmax(scores)
    return weights @ v

def mha(x, attn, n_head, layer_idx , use_kv_cache = False, n_past = 0):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: Complete the code of the multi-head attention
        
        Input: 
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
        Output: Tensorying multi-head attention and linear transformation, shape [n_seq, n_embd].
    """
    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    # qkv projection
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    
    # Split into qkv
    """
        Task: Split the q,k,v matrix from the tensor x
        Notes: [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
    """
    qkv = x.chunk(3, dim = -1) # need to modify

    # Split into heads
    qkv_heads = [qkv_part.chunk(n_head, dim=-1) for qkv_part in qkv]  # 3 * [n_seq, n_embd] -> 3 * n_head * [n_seq, n_embd/n_head]
    qkv_heads = list(zip(*qkv_heads))  # [3, n_head, n_seq, n_embd/n_head]

    n_seq= x.shape[0]
    cache_active = USE_CACHE and layer_idx is not None
    is_prefill = use_kv_cache
    n_k = n_past + n_seq

    new_heads = []
    for h, (q, k, v) in enumerate(qkv_heads):
        if is_prefill:
            k_full, v_full = k, v
        else:
            assert layer_idx is not None
            bucket = KV_CACHE[layer_idx][h]
            k_full = torch.cat(bucket["K"] + [k], dim = 0)
            v_full = torch.cat(bucket["V"] + [v], dim = 0)
        new_heads.append((q, k_full, v_full, k, v))

    # Causal mask to hide future inputs from being attended to
    """
        Task: Construct mask matrix
            Notes: 
                | 0  -inf -inf ... -inf |
                | 0    0  -inf ... -inf |
                | 0    0    0  ... -inf |
                |...  ...  ... ...  ... | 
                | 0    0    0  ...   0  |
        Mask is a tensor whose dimension is [n_seq, n_seq]
    """

    if is_prefill:
        causal_mask = torch.triu(torch.full((n_seq, n_seq), float('-inf')), diagonal=1) # need to modify
    else:
        causal_mask = torch.zeros((n_seq, n_k))

    # Perform attention over each head
    out_heads = [attention(q, k_full, v_full, causal_mask) for q, k_full, v_full, _, _ in new_heads]  # n_head * [n_seq, n_embd/n_head]

    if cache_active:
        assert layer_idx is not None
        if is_prefill:
            for h, (_, _, _, k_new, v_new) in enumerate(new_heads):
                for row in k_new.split(1, dim=0):  # n_seq × [1, d_head]
                    KV_CACHE[layer_idx][h]["K"].append(row)
                for row in v_new.split(1, dim=0):
                    KV_CACHE[layer_idx][h]["V"].append(row)
        else:
            for h, (_, _, _, k_new, v_new) in enumerate(new_heads):
                KV_CACHE[layer_idx][h]["K"].append(k_new)  # [1, d_head]
                KV_CACHE[layer_idx][h]["V"].append(v_new)
    # Merge heads
    """
        Task: merge multi-heads resultsf
        Notes: n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    """
    x = torch.cat(out_heads, dim = -1) # need to modify
    
    # Out projection
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    return x


def transformer_block(x, block, n_head, layer_idx, use_kv_cache, n_past):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']
    
    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head, layer_idx = layer_idx,  use_kv_cache = use_kv_cache, n_past = n_past)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, n_head, cache_length = 0 ):  # [n_seq] -> [n_seq, n_vocab]
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    # token + positional embeddings
    first_pass = (cache_length == 0)
    n_past = 0 if first_pass else cache_len()

    x = wte[inputs] + wpe[range(cache_length, cache_length + len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    
    x = torch.Tensor(x)
    # forward pass through n_layer transformer blocks
    for layer_idx, block in enumerate(blocks):
        x = transformer_block(x, block, n_head=n_head, layer_idx = layer_idx, use_kv_cache = first_pass, n_past = n_past)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm
    global KV_CACHE

    reset_kv_cache()
    KV_CACHE = init_kv_cache(len(params['blocks']), n_head)

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        if not use_cache():
            logits = gpt2(inputs, params, n_head=n_head, cache_length = 0)
        else:
            logits = gpt2([inputs[-1]], params, n_head=n_head, cache_length = cache_len())  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids

def greedy_speculative_generate(inputs, draft_params, target_params, hparams_draft, hparams_target, n_tokens_to_generate, K):
    
    """
        Task: Load 124M and 1558M models at the same time, use greedy sampling, and complete speculative decoding
    
        Inputs:
            inputs (list): The initial list of token IDs from the prompt.
            draft_params, target_params: Model weights for the draft and target models.
            hparams_draft, hparams_target: Hyperparameters for both models.
            n_tokens_to_generate (int): The number of new tokens to generate.
            K (int): The number of tokens the draft model speculates at each step (e.g., 4).

        Returns:
            list: A list of newly generated token IDs.
            
    """
    generated_ids = []
    current_inputs = list(inputs)

    while len(generated_ids) < n_tokens_to_generate:
        pass

    return generated_ids


def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    start = time.time()
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)
