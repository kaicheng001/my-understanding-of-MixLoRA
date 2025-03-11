# <p align="center">My Unified Understanding of MixLoRA</p>
:fire::fire::fire: Last Updated on 2025.03.11 :fire::fire::fire:

## Table of Contents
- [how to initalize LoRA expert and SA LoRA](#how-to-initalize-lora-expert-and-sa-lora)
- [how to compute loss function](#how-to-compute-loss-function)
- [top_k_and_load_balance](#top_k_and_load_balance)
- [modeling_llama.py](#modeling_llamapy)
- [Expert Capacity and Load Balancing](#expert-capacity-and-load-balancing)

## how to initalize LoRA expert and SA LoRA
the code to initalize LoRA expert is in
```
MoE-PEFT/moe_peft/model.py
```

in the function 
```python
def init_lora_layer_weight(
    transformer_layer: LLMDecoder,
    llm_config: LLMModelConfig,
    lora_config: LoraConfig,
    lora_weights: Optional[Dict[str, torch.Tensor]],
):
```




## how to compute loss function
```
MoE-PEFT/moe_peft/model.py
```


## 模型在训练比如组别为1和组别为3的数据每个专家是如何被选择且激活的（路由机制）

classification, Router 


## 模型在训练过程中哪些参数被冻结，哪些参数被解冻

the code to freeze parameters in llm is in
```
MoE-PEFT/moe_peft/model.py
```

in the function 
```python
class LLMModel(torch.nn.Module):


    def from_pretrained(
        name_or_path: str,
        device: str,
        bits: int = None,
        attn_impl: str = "eager",
        use_sliding_window: bool = False,
        load_dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype = torch.bfloat16,
        double_quant: bool = True,
        quant_type: str = "nf4",
    ) -> "LLMModel":
```


## Expert Capacity and Load Balancing

