# FlexGen 推理流程详解

## 📖 概述

FlexGen 是一个高吞吐量的大语言模型推理引擎，其核心思想是通过在 **GPU、CPU 和磁盘** 之间智能地分配和调度模型权重、KV缓存和激活值，在有限的GPU内存下实现高吞吐量推理。

## 🏗️ 核心架构

```
用户请求 → OptLM → 分层执行 → 输出
                ↓
        ExecutionEnv (GPU/CPU/Disk)
                ↓
        Policy (内存分配策略)
                ↓
        Layers (InputEmbed + Transformer + OutputEmbed)
```

## 🔄 完整推理流程

### 第一阶段：初始化 (Initialization)

#### 1. 环境创建
```python
# apps/completion.py
env = ExecutionEnv.create(offload_dir)
```

**ExecutionEnv** 包含三类设备：
- **GPU** (`TorchDevice("cuda:0")`) - 用于高速计算
- **CPU** (`TorchDevice("cpu")`) - 中间缓存
- **Disk** (`TorchDisk`) - 大容量存储
- **Mixed** (`TorchMixedDevice`) - 混合设备，跨设备分布

#### 2. 策略定义
```python
policy = Policy(
    gpu_batch_size=32,         # GPU批次大小
    num_gpu_batches=1,         # GPU批次数量
    w_gpu_percent=100,         # 权重在GPU的百分比
    w_cpu_percent=0,           # 权重在CPU的百分比
    cache_gpu_percent=100,     # KV缓存在GPU的百分比
    cache_cpu_percent=0,       # KV缓存在CPU的百分比
    act_gpu_percent=100,       # 激活值在GPU的百分比
    act_cpu_percent=0,         # 激活值在CPU的百分比
    overlap=True,              # I/O与计算重叠
    sep_layer=True,            # 分离注意力和MLP层
    compress_weight=False,     # 权重压缩
    compress_cache=False,      # KV缓存压缩
)
```

**Policy** 决定了：
- 数据在不同设备间的分配比例
- 是否启用压缩（4-bit量化）
- 是否重叠I/O和计算

#### 3. 模型初始化
```python
model = OptLM(model_name, env, path, policy)
```

**OptLM 构造过程：**
```python
class OptLM:
    def __init__(self, config, env, path, policy):
        # 1. 创建模型层
        layers = [
            InputEmbed(...),                    # 输入嵌入层
            SelfAttention(...) × N,             # N个自注意力层
            MLP(...) × N,                       # N个MLP层
            OutputEmbed(...),                   # 输出层
        ]
        
        # 2. 创建CUDA流（用于异步执行）
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()
        
        # 3. 分配缓冲区
        self.cache_home[j][k]       # 存储KV缓存
        self.cache_read_buf[j][k]   # 读缓冲
        self.cache_write_buf[j][k]  # 写缓冲
        self.weight_read_buf[j]     # 权重缓冲
        self.hidden[i][j][k]        # 隐藏状态
        
        # 4. 初始化权重
        self.init_all_weights()
```

#### 4. 权重加载策略
```python
def init_weight_list(weight_specs, policy, env):
    # 根据policy决定每个权重张量存放的位置
    for spec in weight_specs:
        # 计算该权重应该放在哪个设备
        home = get_choice(percent, [disk, cpu, gpu])
        
        if compress:
            # 使用压缩存储（4-bit量化）
            weight = home.compressed_device.allocate(...)
        else:
            # 正常存储（FP16/FP32）
            weight = home.allocate(...)
        
        # 从文件加载权重
        weight.load_from_np_file(filename)
```

### 第二阶段：生成 (Generation)

#### 1. 任务创建
```python
output_ids = model.generate(
    inputs=input_ids,          # 输入token ids
    max_new_tokens=32,         # 最大生成长度
    do_sample=True,            # 采样模式
    temperature=0.7,           # 温度
    stop=stop_token,           # 停止符
)
```

**Task 对象：**
```python
task = Task(
    inputs=inputs,             # 输入序列
    prompt_len=len(inputs[0]), # 提示词长度
    gen_len=max_new_tokens,    # 生成长度
    do_sample=do_sample,
    temperature=temperature,
    stop=stop,
)
```

#### 2. 缓存初始化
```python
# 为每一层、每个GPU批次初始化KV缓存
for j in range(num_layers):
    for k in range(num_gpu_batches):
        self.init_cache(j, k)
```

KV缓存的形状：
- **Key**: `(max_seq_len, batch_size * num_heads, head_dim)`
- **Value**: `(max_seq_len, batch_size * num_heads, head_dim)`

#### 3. 生成循环 (核心)

FlexGen 有三种生成模式：

##### 模式A: 无重叠 (Normal)
```python
def generation_loop_normal():
    for i in range(gen_len):              # 对每个token位置
        for k in range(num_gpu_batches):  # 对每个GPU批次
            update_attention_mask(i, k)
        
        for j in range(num_layers):       # 对每一层
            for k in range(num_gpu_batches):
                # 步骤1: 加载权重
                load_weight(i, j, k)
                
                # 步骤2: 加载KV缓存
                load_cache(i, j, k)
                
                # 步骤3: 加载隐藏状态
                load_hidden(i, j, k)
                
                # 步骤4: 计算
                compute_layer(i, j, k)
                
                # 步骤5: 存储隐藏状态
                store_hidden(i, j, k)
                
                # 步骤6: 存储KV缓存
                store_cache(i, j, k)
```

##### 模式B: 单批次重叠 (Overlap Single Batch) ⚡
```python
def generation_loop_overlap_single_batch():
    # 预加载第一层权重
    load_weight(0, 0, 0)
    sync()
    
    for i in range(gen_len):
        update_attention_mask(i, 0)
        
        for j in range(num_layers):
            # 🔥 并行执行：
            load_weight(i, j+1, 0)      # 预加载下一层权重
            load_cache(i, j+1, 0)       # 预加载下一层缓存
            load_hidden(i, j, 0)        # 加载当前隐藏状态
            compute_layer(i, j, 0)      # 当前层计算
            store_cache(i, j-1, 0)      # 存储上一层缓存
            store_hidden(i, j, 0)       # 存储当前隐藏状态
            sync()
```

**关键优化：** 使用流水线技术，在计算第 j 层时，同时：
- 预加载第 j+1 层的数据
- 存储第 j-1 层的结果

##### 模式C: 多批次重叠 (Overlap Multi Batch) ⚡⚡
```python
def generation_loop_overlap_multi_batch():
    # 预加载
    for k in range(num_gpu_batches):
        load_weight(0, 0, k)
    load_hidden(0, 0, 0)
    sync()
    
    for i in range(gen_len):
        for k in range(num_gpu_batches):
            update_attention_mask(i, k)
        
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                # 🚀 最大化并行：
                load_weight(i, j+1, k)      # 预加载下一层权重
                load_cache(i, j, k+1)       # 预加载下一批缓存
                store_hidden(i, j, k-1)     # 存储上一批隐藏状态
                load_hidden(i, j, k+1)      # 预加载下一批隐藏状态
                compute_layer(i, j, k)      # 当前计算
                store_cache(i, j, k-1)      # 存储上一批缓存
                sync()
```

**关键优化：** 在多个GPU批次间流水线执行，最大化设备利用率。

#### 4. 详细步骤解析

##### 步骤1: 加载权重 (load_weight)
```python
def load_weight(self, i, j, k):
    # 使用专用CUDA流异步加载
    with torch.cuda.stream(self.load_weight_stream):
        # 从weight_home加载到weight_read_buf
        # 如果启用压缩，则解压
        self.layers[j].load_weight(
            self.weight_home[j], 
            self.weight_read_buf[j], 
            k
        )
```

**权重流动：**
```
Disk/CPU/GPU (weight_home) 
    ↓ smart_copy (异步)
GPU (weight_read_buf)
    ↓
计算单元
```

##### 步骤2: 加载KV缓存 (load_cache)
```python
def load_cache(self, i, j, k):
    if i == 0:  # prefill阶段，无历史缓存
        return
    
    with torch.cuda.stream(self.load_cache_stream):
        # 加载历史K, V
        indices = (slice(0, prompt_len + i), slice(0, batch_size))
        k_cache = k_home.smart_copy(gpu, indices)
        v_cache = v_home.smart_copy(gpu, indices)
```

**缓存流动：**
```
存储设备 (cache_home)
    ↓ 加载已生成的K, V
GPU (cache_read_buf)
    ↓
注意力计算
    ↓ 追加新的K, V
GPU (cache_write_buf)
    ↓ 
存储设备 (cache_home)
```

##### 步骤3: 计算层 (compute_layer)

**以自注意力层为例：**
```python
def forward(self, hidden, cache_read_buf, weight_read_buf, 
            attention_mask, cache_write_buf, i, k):
    
    # 1. LayerNorm
    h = layer_norm(hidden, w_ln, b_ln)
    
    # 2. 计算 Q, K, V
    q = linear(h, w_q, b_q)
    k = linear(h, w_k, b_k)
    v = linear(h, w_v, b_v)
    
    # 3. 拼接历史K, V (仅在decode阶段)
    if i > 0:
        k_cache, v_cache = cache_read_buf.val
        k = concat([k_cache, k], dim=0)  # 拼接序列维度
        v = concat([v_cache, v], dim=0)
    
    # 4. 计算注意力
    # Q: (1, b*n_head, head_dim)  [当前token]
    # K: (seq_len, b*n_head, head_dim)  [所有历史token]
    # V: (seq_len, b*n_head, head_dim)
    scores = matmul(q, k.transpose()) / sqrt(head_dim)
    scores = scores + attention_mask
    attn_weights = softmax(scores, dim=-1)
    
    # 稀疏注意力优化（可选）
    if attn_sparsity < 1.0:
        attn_weights = top_k_sparsify(attn_weights)
    
    context = matmul(attn_weights, v)
    
    # 5. 输出投影
    output = linear(context, w_out, b_out)
    
    # 6. 残差连接
    hidden.val = output + hidden.val
    
    # 7. 存储新的K, V到write_buf
    cache_write_buf.store((k[-1:], v[-1:]))  # 只存储新token
```

**MLP层：**
```python
def forward(self, hidden, ...):
    # 1. LayerNorm
    h = layer_norm(hidden, w_ln, b_ln)
    
    # 2. 两层全连接 + 激活
    h = linear(h, w_fc1, b_fc1)
    h = relu(h)
    h = linear(h, w_fc2, b_fc2)
    
    # 3. 残差连接
    hidden.val = h + hidden.val
```

##### 步骤4: 存储缓存 (store_cache)
```python
def store_cache(self, i, j, k):
    if i == gen_len - 1:  # 最后一个token，无需存储
        return
    
    with torch.cuda.stream(self.store_cache_stream):
        # 将cache_write_buf写回cache_home
        k_new, v_new = cache_write_buf.pop()
        
        # 追加到现有缓存
        pos = prompt_len + i
        cache_home[pos] = (k_new, v_new)
```

##### 步骤5: 输出处理 (store_hidden)
```python
def store_hidden(self, i, j, k):
    if j == num_layers - 1:  # 最后一层
        # 转换为token id
        logits = hidden.val
        if do_sample:
            # 温度采样
            probs = softmax(logits / temperature)
            token_id = sample(probs)
        else:
            # 贪心解码
            token_id = argmax(logits)
        
        # 存储到输出
        output_ids[batch_idx, prompt_len + i] = token_id
        
        # 检查停止条件
        if token_id == stop_token:
            stopped[batch_idx] = True
```

### 第三阶段：清理 (Cleanup)

```python
# 删除所有缓存
for j in range(num_layers):
    for k in range(num_gpu_batches):
        delete_cache(j, k)

# 关闭I/O线程
env.close_copy_threads()

return output_ids
```

## 🎯 核心优化技术

### 1. 分层卸载 (Offloading)
```
模型层次                设备分配
输入嵌入层     →     GPU 100%
Transformer层0  →     GPU 80% + CPU 20%
Transformer层1  →     GPU 60% + CPU 30% + Disk 10%
...
输出层         →     GPU 100%
```

### 2. 压缩存储 (Compression)
```python
# 4-bit分组量化
compressed_weight = quantize(
    weight, 
    num_bits=4,      # 4位存储
    group_size=64,   # 64个元素一组
    symmetric=False  # 非对称量化
)

# 存储格式：(quantized_data, scale, zero_point)
# 内存节约：16bit → 4bit = 75% 减少
```

### 3. 流水线并行 (Pipelining)
```
时间轴：
|----加载L1----|
              |----加载L2----|
                            |----加载L3----|
       |----计算L1----|
                     |----计算L2----|
                                   |----计算L3----|
              |----存储L1----|
                            |----存储L2----|
```

### 4. 异步拷贝 (Async Copy)
```python
# 使用多个CUDA流并行执行
stream1: load_weight(layer_j+1)    # 预加载
stream2: load_cache(layer_j+1)     # 预加载
stream3: store_cache(layer_j-1)    # 后台存储
default: compute_layer(layer_j)    # 计算
```

### 5. 注意力稀疏化 (Attention Sparsity)
```python
# Top-K稀疏注意力
if attn_sparsity < 1.0:
    k = int(seq_len * attn_sparsity)
    # 只保留top-k个注意力权重
    attn_weights = top_k(attn_weights, k)
    # 减少访存和计算
```

## 📊 执行时序

### Prefill 阶段 (i=0)
```
输入：完整的prompt (seq_len=128)
输出：第一个新token

1. InputEmbed: (batch, 128) → (batch, 128, hidden_dim)
2. 每个Transformer层:
   - Self-Attention: 计算所有prompt tokens之间的注意力
   - MLP: 前馈网络
   - 存储K, V cache: (128, batch*n_head, head_dim)
3. OutputEmbed: → 生成第1个token
```

### Decode 阶段 (i=1, 2, ..., gen_len-1)
```
输入：上一个生成的token (seq_len=1)
输出：下一个新token

1. InputEmbed: (batch, 1) → (batch, 1, hidden_dim)
2. 每个Transformer层:
   - 加载历史K, V: (128+i, ...)
   - Self-Attention: 新token attend to 所有历史tokens
   - 追加新的K, V到cache: (128+i+1, ...)
   - MLP
3. OutputEmbed: → 生成第i+1个token
```

**关键差异：**
- **Prefill**: 计算密集，矩阵大，一次处理多个token
- **Decode**: 访存密集，矩阵小，每次只处理1个token

## 🔍 内存使用分析

```python
# 以OPT-175B, batch_size=32, seq_len=512为例

# 权重 (~350GB FP16)
weights = 175B_params × 2_bytes = 350GB

# KV缓存 (每层)
n_layers = 96
hidden_dim = 12288
n_heads = 96
head_dim = 128
cache_per_layer = 2 × seq_len × batch × n_heads × head_dim × 2_bytes
                = 2 × 512 × 32 × 96 × 128 × 2
                = 800MB per layer
total_cache = 800MB × 96 = 76.8GB

# 激活值 (前向传播)
act_per_token ≈ 12 × hidden_dim^2 × batch
              ≈ 12 × 12288^2 × 32 × 2_bytes
              ≈ 100GB

# FlexGen策略示例 (40GB GPU)
GPU: 权重10% + 缓存50% + 激活100% ≈ 35GB + 38GB + 100GB → 需要offload
策略: 权重→CPU, 缓存→GPU+CPU, 激活→GPU, 流水线执行
```

## 🚀 性能优势

**FlexGen vs 标准推理:**

| 维度 | 标准推理 | FlexGen |
|------|---------|---------|
| GPU内存需求 | 完整模型 (350GB) | 部分模型 (40GB) |
| Batch Size | 小 (1-4) | 大 (32-512) |
| 吞吐量 | 低 | 高 (10-100x) |
| 延迟 | 低 | 中高 |
| 适用场景 | 交互式对话 | 批量处理 |

## 💡 使用建议

1. **大batch高吞吐**: `gpu_batch_size=128`, `overlap=True`
2. **内存受限**: 调整percent参数，offload到CPU/Disk
3. **延迟敏感**: 减小batch_size，增加GPU内存比例
4. **压缩权重**: `compress_weight=True` 节省75%内存
5. **稀疏注意力**: `attn_sparsity=0.5` 减少长序列开销

## 📚 相关文件

- **核心引擎**: [flex_opt.py](flexllmgen/flex_opt.py)
- **后端实现**: [pytorch_backend.py](flexllmgen/pytorch_backend.py)
- **压缩实现**: [compression.py](flexllmgen/compression.py)
- **应用示例**: [apps/completion.py](flexllmgen/apps/completion.py)
- **分布式**: [dist_flex_opt.py](flexllmgen/dist_flex_opt.py)
