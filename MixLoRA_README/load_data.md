```
MoE-PEFT/moe_peft/dispatcher.py
```


你的数据加载到模型中的流程分为以下几个关键步骤：

---

### 1. **数据集加载 (`load_dataset`)**
```python
def load_dataset(data_path: str):
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        return datasets.load_dataset("json", data_files=data_path)
    else:
        if ":" in data_path:
            result = data_path.split(":")
            return datasets.load_dataset(result[0], result[1])
        else:
            return datasets.load_dataset(data_path)
```
- **功能**：根据路径加载原始数据集。
- **支持的格式**：
  - **JSON/JSONL**：直接加载本地文件（如 `data.json`）。
  - **HuggingFace 数据集**：通过 `dataset_name` 或 `dataset_name:subset` 加载（如 `glue:sst2`）。

---

### 2. **数据预处理与分词 (`TrainTask.load_data`)**
```python
def load_data(self):
    self.train_token_data_ = self.dataload_function_(self.tokenizer_)
    ...
```
- **核心逻辑**：
  1. **调用自定义的 `dataload_function_`**：  
     该函数由用户提供，负责将原始数据集转换为模型输入格式（如 `InputData` 对象列表）。  
     - 输入：`tokenizer_`（分词器）。
     - 输出：`train_token_data_`（处理后的训练数据，每个元素包含 `tokens` 等字段）。

  2. **截断与长度统计**：  
     - 截断超过 `train_cutoff_len_` 的序列。
     - 记录最大序列长度（用于后续填充或日志）。

  3. **数据排序或打乱**：  
     - **按长度排序**（`group_by_length_=True`）：优化填充效率，减少计算浪费。  
     - **随机打乱**（`group_by_length_=False`）：增加训练随机性。

---

### 3. **数据分批与迭代 (`TrainTask.get_train_data`)**
```python
def get_train_data(self) -> List[InputData]:
    start_idx = self.next_train_data_start_idx_
    end_idx = start_idx + self.max_train_micro_batch_size_
    ret_data = self.train_token_data_[start_idx:end_idx]
    ...
    self.next_train_data_start_idx_ += self.max_train_micro_batch_size_
    ...
    return ret_data
```
- **分批策略**：
  - **微批次（Micro Batch）**：每次返回 `max_train_micro_batch_size_` 条数据。
  - **索引管理**：通过 `next_train_data_start_idx_` 跟踪当前批次的起始位置。
  - **Epoch 管理**：当索引超出数据长度时，重置索引并递增 `epoch_cnt_`。

- **输出格式**：  
  返回 `List[InputData]`，每个 `InputData` 对象包含分词后的 `tokens`（模型的直接输入）。

---

### 4. **数据传递到模型**
在训练循环中，数据通过以下步骤传递到模型：
1. **获取批次数据**：调用 `get_train_data()` 获取当前批次的 `InputData`。
2. **填充（Padding）**：  
   将 `tokens` 填充到相同长度（通常在 `dataload_function_` 或模型输入处理时完成）。
3. **模型前向传播**：  
   将填充后的张量（如 `input_ids`, `attention_mask`）输入模型。

---

### 关键细节总结
| **步骤**           | **实现位置**               | **核心功能**                                                                 |
|--------------------|---------------------------|----------------------------------------------------------------------------|
| 数据集加载         | `load_dataset` 函数         | 根据路径加载原始数据（JSON/HF 数据集）                                      |
| 数据预处理与分词   | `TrainTask.load_data`      | 调用 `dataload_function_` 生成 `InputData`，截断、排序或打乱数据            |
| 数据分批           | `TrainTask.get_train_data` | 按 `max_train_micro_batch_size_` 分批，管理 Epoch 和索引                    |
| 模型输入           | 训练循环中                 | 将 `InputData` 转换为张量（如 `input_ids`），传递给模型前向传播             |

---

### 示例流程
假设 `dataload_function_` 的实现如下：
```python
def my_dataload_function(tokenizer):
    # 1. 加载原始数据（如 JSON）
    raw_data = [{"text": "Hello world!"}, ...]
    # 2. 分词并转换为 InputData
    token_data = []
    for example in raw_data:
        tokens = tokenizer.encode(example["text"])
        token_data.append(InputData(tokens=tokens))
    return token_data
```
则整体流程为：
```
原始数据 → 分词 → InputData → 分批 → 模型输入张量
```