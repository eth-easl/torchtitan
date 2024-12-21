import torch
from typing import Dict, Any
from transformers import LlamaConfig, AutoModelForCausalLM, AutoTokenizer
from torchtitan.models.llama import Transformer
from torchtitan.models.llama.model import ModelArgs
def permute(w, n_heads, dim1, dim2):
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

def build_model(config):
    model_cls = Transformer
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config = ModelArgs(
        dim=config["hidden_size"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
        vocab_size=config["vocab_size"],
    )
    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)
    return model

def to_huggingface(states: Dict[str, Any], export_dtype: str, config: Dict):
    # step 1: create a config file containing the model architecture
    config = LlamaConfig(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_dim"],
        num_hidden_layers=config["n_layers"],
        num_attention_heads=config["n_heads"],
        num_key_value_heads=config["n_kv_heads"],
        hidden_act=config["activation"],
        max_position_embeddings=config["max_seq_len"],
        initializer_range=0.02,
        rope_theta=config["rope_theta"],
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    # step 2: load the model weights, we need some translation here
    new_state_dict = {}
    with torch.inference_mode():
        states_dicts = states["model"]
        states_dicts = {k: v.contiguous() for k, v in states_dicts.items()}
        # step 2.1: handle non-transformer-blocks
        new_state_dict["model.embed_tokens.weight"] = states_dicts["tok_embeddings.weight"]
        new_state_dict["model.norm.weight"] = states_dicts["norm.weight"]
        new_state_dict["lm_head.weight"] = states_dicts["output.weight"]
        dims_per_head = config.hidden_size // config.num_attention_heads

        # step 2.2: handle transformer blocks
        for i in range(config.num_hidden_layers):

            new_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = permute(
                w=states_dicts[f"layers.{i}.attention.wq.weight"],
                n_heads=config.num_attention_heads,
                dim1=config.hidden_size,
                dim2=config.hidden_size,
            )

            new_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = permute(
                w=states_dicts[f"layers.{i}.attention.wk.weight"],
                n_heads=config.num_key_value_heads,
                dim1=dims_per_head * config.num_key_value_heads,
                dim2=config.hidden_size,
            )

            new_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = states_dicts[f"layers.{i}.attention.wv.weight"]
            new_state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = states_dicts[
                f"layers.{i}.attention.wo.weight"
            ]
            new_state_dict[f"model.layers.{i}.input_layernorm.weight"] = states_dicts[
                f"layers.{i}.attention_norm.weight"
            ]
            new_state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = states_dicts[
                f"layers.{i}.ffn_norm.weight"
            ]
            new_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = states_dicts[f"layers.{i}.feed_forward.w1.weight"]
            new_state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = states_dicts[f"layers.{i}.feed_forward.w2.weight"]
            new_state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = states_dicts[f"layers.{i}.feed_forward.w3.weight"]
        
        new_state_dict = {k: v.to(torch.bfloat16) for k, v in new_state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True, assign=True)
        model.eval()
        return model, config

if __name__ =="__main__":
    model_config = {
        "architecture":"llama",
        "activation" : "silu",
        "hidden_size" : 768,
        "initializer_range" : 0.02,
        "intermediate_dim": 2816,
        "max_seq_len": 1024, # equal to sequence length
        "n_heads": 12,
        "n_layers": 12,
        "n_kv_heads": 12,
        "norm_eps": 1.0e-05,
        "vocab_size": 50432,
        "rope_theta": 10000,
    }
    print(f"Building model...")
    model = build_model(model_config)
    model.to_empty(device="cpu")
    print(f"Loading ckpt...")
    state_dict = torch.load(
        "data/50000.pt", 
        map_location=torch.device("cpu"), 
        weights_only=False
    )
    model, hf_config = to_huggingface(state_dict, "bfloat16", model_config)
    print(f"Write to disk...")
    model.save_pretrained("data/mixtera")
    hf_config.save_pretrained("data/mixtera")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.save_pretrained("data/mixtera")