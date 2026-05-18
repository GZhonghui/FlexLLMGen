# FlexLLMGen 代码地图 - 入口与核心代码

## 📍 入口代码 (Entry Points)

### 1. 主入口 - 命令行直接运行
**位置**: [flexllmgen/flex_opt.py](flexllmgen/flex_opt.py#L1323)

```bash
# 直接作为模块运行
python3 -m flexllmgen.flex_opt \
    --model facebook/opt-1.3b \
    --gpu-batch-size 32 \
    --percent 100 0 100 0 100 0
```

**入口函数**: `run_flexllmgen(args)` (第1182行)
- 初始化环境和策略
- 创建OptLM模型
- 执行推理和benchmark
- 输出日志

---

### 2. 应用层入口

#### A. 文本补全应用
**位置**: [flexllmgen/apps/completion.py](flexllmgen/apps/completion.py#L71)

```bash
python3 -m flexllmgen.apps.completion \
    --model facebook/opt-6.7b \
    --compress-weight
```

**用途**: 交互式文本生成示例
- 简单的prompt补全
- 演示基本使用方法

#### B. HELM基准测试
**位置**: [flexllmgen/apps/helm_run.py](flexllmgen/apps/helm_run.py#L388)

```bash
python3 -m flexllmgen.apps.helm_run \
    --task helm \
    --model facebook/opt-30b
```

**用途**: 运行HELM标准基准测试
- 支持多种NLP任务
- 批量评估模型性能

#### C. 数据处理应用
**位置**: [flexllmgen/apps/data_wrangle/data_wrangle_run.py](flexllmgen/apps/data_wrangle/data_wrangle_run.py#L542)

**用途**: 大规模数据清洗和标注

---

### 3. 分布式入口
**位置**: [flexllmgen/dist_flex_opt.py](flexllmgen/dist_flex_opt.py#L666)

```bash
# 多节点/多GPU分布式推理
torchrun --nproc_per_node=4 -m flexllmgen.dist_flex_opt \
    --model facebook/opt-175b
```

---

### 4. Benchmark套件
**位置**: [benchmark/flexllmgen/bench_suite.py](benchmark/flexllmgen/bench_suite.py#L179)

```bash
python3 benchmark/flexllmgen/bench_suite.py \
    --model facebook/opt-175b
```

**用途**: 系统性能测试和对比

---

## 🎯 核心代码 (Core Implementation)

### 核心文件层次结构

```
flexllmgen/
├── 🔴 flex_opt.py           ← 核心引擎 (1331行)
├── 🔴 pytorch_backend.py    ← 后端实现 (907行)
├── 🟡 compression.py         ← 压缩算法 (361行)
├── 🟡 dist_utils.py          ← 分布式工具
├── 🟡 dist_flex_opt.py       ← 分布式引擎
├── 🟢 opt_config.py          ← 模型配置
├── 🟢 utils.py               ← 工具函数
└── 🟢 timer.py               ← 性能计时
```

---

## 🔴 核心代码详解

### 1. flex_opt.py - 推理引擎核心 ⭐⭐⭐⭐⭐
**位置**: [flexllmgen/flex_opt.py](flexllmgen/flex_opt.py)

#### 关键类和函数

##### `class OptLM` (第582行) - 主模型类
```python
class OptLM:
    def __init__(self, config, env, path, policy):
        """
        核心模型类，管理整个推理流程
        - 初始化所有层
        - 管理权重和缓存
        - 协调I/O和计算
        """
    
    def generate(self, inputs, max_new_tokens=32, ...):
        """
        【最核心】生成函数 (第825行)
        - 接收输入token ids
        - 执行自回归生成
        - 返回输出token ids
        """
        
    def generation_loop_overlap_multi_batch(self):
        """
        多批次流水线生成循环 (第1034行)
        - 最高性能的执行模式
        - I/O与计算完全重叠
        """
```

##### `class InputEmbed` (第135行) - 输入嵌入层
```python
class InputEmbed:
    def init_weight(self, weight_home, path):
        """加载embedding权重"""
    
    def forward(self, hidden, ...):
        """token_id → embedding向量"""
```

##### `class SelfAttention` (第268行) - 自注意力层
```python
class SelfAttention:
    def forward(self, hidden, cache_read_buf, ...):
        """
        【计算密集】自注意力机制
        - 计算Q, K, V
        - 注意力分数计算
        - KV缓存管理
        """
    
    def load_cache(self, cache_home, cache_read_buf, i):
        """加载历史KV缓存 (第327行)"""
    
    def store_cache(self, cache_home, cache_write_buf, i):
        """存储新的KV缓存 (第398行)"""
```

##### `class MLP` (第439行) - 前馈网络层
```python
class MLP:
    def forward(self, hidden, ...):
        """
        两层全连接网络
        - FC1: h → 4h (扩展)
        - ReLU激活
        - FC2: 4h → h (压缩)
        """
```

##### `class OutputEmbed` (第219行) - 输出层
```python
class OutputEmbed:
    def forward(self, hidden, ...):
        """
        hidden → logits → token_id
        - 支持采样和贪心解码
        """
```

##### 关键辅助函数

```python
def init_weight_list(weight_specs, policy, env):
    """
    【内存管理核心】权重初始化 (第88行)
    - 根据policy决定权重放置位置
    - 支持压缩存储
    - 智能设备分配
    """

def load_weight(self, i, j, k, overlap=True):
    """异步加载权重 (第656行)"""

def load_cache(self, i, j, k, overlap=True):
    """异步加载KV缓存 (第685行)"""

def compute_layer(self, i, j, k):
    """执行层计算 (第790行)"""
```

---

### 2. pytorch_backend.py - 后端实现 ⭐⭐⭐⭐⭐
**位置**: [flexllmgen/pytorch_backend.py](flexllmgen/pytorch_backend.py)

#### 关键类

##### `class TorchTensor` (第63行)
```python
class TorchTensor:
    """
    【张量抽象】统一的张量表示
    - 支持GPU/CPU/Disk/Mixed/Compressed
    - 异步数据拷贝
    - 智能内存管理
    """
    
    def smart_copy(self, dst, indices=None):
        """
        智能拷贝 (第171行)
        - 自动选择最优拷贝路径
        - 支持异步传输
        """
    
    def load_from_np_file(self, filename):
        """从numpy文件加载 (第148行)"""
```

##### `class TorchDevice` (第315行)
```python
class TorchDevice:
    """
    【设备抽象】GPU/CPU设备管理
    - 内存分配和追踪
    - 张量生命周期管理
    """
    
    def allocate(self, shape, dtype, pin_memory=False):
        """分配张量内存 (第346行)"""
    
    def opt_attention(self, ...):
        """
        【计算核心】注意力计算 (第490行)
        - 支持稀疏注意力
        - 优化的矩阵运算
        """
    
    def opt_gen_mhsa(self, ...):
        """
        【解码阶段】多头自注意力 (第556行)
        - 优化的KV缓存访问
        """
    
    def mha_gen_mlp(self, ...):
        """MLP前馈网络计算 (第627行)"""
```

##### `class TorchDisk` (第714行)
```python
class TorchDisk:
    """
    【磁盘I/O】后台异步磁盘读写
    - 多线程I/O
    - 避免阻塞计算
    """
    
    def allocate(self, shape, dtype, ...):
        """分配磁盘存储空间"""
    
    def copy_worker(self):
        """后台I/O工作线程 (第742行)"""
```

##### `class TorchMixedDevice` (第840行)
```python
class TorchMixedDevice:
    """
    【混合设备】跨设备分布数据
    - GPU + CPU + Disk组合
    - 自动分片管理
    """
```

---

### 3. compression.py - 压缩算法 ⭐⭐⭐⭐
**位置**: [flexllmgen/compression.py](flexllmgen/compression.py)

#### 关键实现

```python
@dataclasses.dataclass
class CompressionConfig:
    """
    压缩配置 (第29行)
    - num_bits: 量化位数 (通常4-bit)
    - group_size: 分组大小 (64)
    - symmetric: 对称/非对称量化
    """

class TorchCompressedDevice:
    """
    【压缩存储】压缩设备包装器 (第46行)
    - 4-bit分组量化
    - 75%内存节省
    """
    
    def allocate(self, shape, dtype, config, ...):
        """分配压缩张量 (第72行)"""

def general_copy_compressed(dst, src, ...):
    """
    压缩张量拷贝 (第183行)
    - 自动压缩/解压
    - 支持设备间传输
    """

def compress_int4(data, config):
    """
    【核心算法】4-bit量化压缩 (第253行)
    - 分组量化
    - 计算scale和zero_point
    """

def decompress_int4(data, scale, config):
    """4-bit解压缩 (第285行)"""
```

---

### 4. dist_flex_opt.py - 分布式引擎 ⭐⭐⭐
**位置**: [flexllmgen/dist_flex_opt.py](flexllmgen/dist_flex_opt.py)

```python
class DistOptLM:
    """
    分布式模型类
    - 张量并行 (Tensor Parallelism)
    - 流水线并行 (Pipeline Parallelism)
    - 跨节点通信
    """
```

---

## 🔥 执行流程中的关键代码路径

### 路径1: 初始化
```
run_flexllmgen()                        [flex_opt.py:1182]
  └─> OptLM.__init__()                  [flex_opt.py:582]
        ├─> init_all_weights()          [flex_opt.py:799]
        │     └─> init_weight_list()    [flex_opt.py:88]
        │           └─> TorchDevice.allocate()  [pytorch_backend.py:346]
        └─> layers创建
              ├─> InputEmbed()          [flex_opt.py:135]
              ├─> SelfAttention()       [flex_opt.py:268]
              ├─> MLP()                 [flex_opt.py:439]
              └─> OutputEmbed()         [flex_opt.py:219]
```

### 路径2: 生成循环
```
OptLM.generate()                                [flex_opt.py:825]
  └─> generation_loop_overlap_multi_batch()     [flex_opt.py:1034]
        └─> for each token position:
              ├─> load_weight()                 [flex_opt.py:656]
              │     └─> TorchTensor.smart_copy() [pytorch_backend.py:171]
              ├─> load_cache()                  [flex_opt.py:685]
              ├─> compute_layer()               [flex_opt.py:790]
              │     └─> Layer.forward()
              │           ├─> TorchDevice.opt_attention()  [pytorch_backend.py:490]
              │           └─> TorchDevice.mha_gen_mlp()    [pytorch_backend.py:627]
              └─> store_cache()                 [flex_opt.py:700]
```

### 路径3: 注意力计算
```
SelfAttention.forward()                         [flex_opt.py:409]
  └─> TorchDevice.opt_attention()               [pytorch_backend.py:490]
        ├─> linear(h, w_q, b_q)  # Q投影
        ├─> linear(h, w_k, b_k)  # K投影
        ├─> linear(h, w_v, b_v)  # V投影
        ├─> opt_gen_mhsa(q, k, v, mask)         [pytorch_backend.py:556]
        │     ├─> scores = q @ k^T / sqrt(d)
        │     ├─> attn = softmax(scores + mask)
        │     └─> out = attn @ v
        └─> linear(out, w_out, b_out)  # 输出投影
```

---

## 📊 代码复杂度统计

| 文件 | 行数 | 核心程度 | 作用 |
|------|------|----------|------|
| flex_opt.py | 1331 | ⭐⭐⭐⭐⭐ | 推理引擎主逻辑 |
| pytorch_backend.py | 907 | ⭐⭐⭐⭐⭐ | 底层计算和I/O |
| compression.py | 361 | ⭐⭐⭐⭐ | 压缩算法 |
| dist_flex_opt.py | 666 | ⭐⭐⭐ | 分布式支持 |
| opt_config.py | 257 | ⭐⭐ | 模型配置 |
| utils.py | 303 | ⭐⭐ | 工具函数 |

---

## 🎓 学习路线建议

### 初学者路线
1. **入口**: [apps/completion.py](flexllmgen/apps/completion.py) (100行) - 最简单的使用示例
2. **配置**: [opt_config.py](flexllmgen/opt_config.py) (257行) - 理解模型配置
3. **工具**: [utils.py](flexllmgen/utils.py) (303行) - 基础数据结构

### 进阶路线
4. **主引擎**: [flex_opt.py](flexllmgen/flex_opt.py) (1331行)
   - 先看 `OptLM.__init__()` (第582行)
   - 再看 `generate()` (第825行)
   - 重点看 `generation_loop_overlap_multi_batch()` (第1034行)

5. **层实现**: [flex_opt.py](flexllmgen/flex_opt.py)
   - `SelfAttention` (第268行)
   - `MLP` (第439行)

### 高级路线
6. **后端**: [pytorch_backend.py](flexllmgen/pytorch_backend.py) (907行)
   - `TorchTensor` (第63行) - 张量抽象
   - `TorchDevice.opt_attention()` (第490行) - 注意力计算
   - `TorchDisk` (第714行) - 异步I/O

7. **优化**: [compression.py](flexllmgen/compression.py) (361行)
   - `compress_int4()` (第253行) - 量化算法

8. **分布式**: [dist_flex_opt.py](flexllmgen/dist_flex_opt.py) (666行)

---

## 🔍 快速定位代码

### 想了解某个功能？直接跳转：

| 功能 | 文件 | 行号 |
|------|------|------|
| 生成主循环 | flex_opt.py | L825-L905 |
| 注意力计算 | pytorch_backend.py | L490-L555 |
| KV缓存管理 | flex_opt.py | L327-L398 |
| 权重加载 | flex_opt.py | L88-L130 |
| 4-bit压缩 | compression.py | L253-L283 |
| 异步I/O | pytorch_backend.py | L714-L839 |
| 分布式通信 | dist_flex_opt.py | L1-L666 |
| 命令行入口 | flex_opt.py | L1323-L1331 |

---

## 💡 总结

### 最重要的3个文件
1. **[flex_opt.py](flexllmgen/flex_opt.py)** - 看这个文件就能理解整个推理流程
2. **[pytorch_backend.py](flexllmgen/pytorch_backend.py)** - 理解底层如何实现
3. **[compression.py](flexllmgen/compression.py)** - 理解内存优化技术

### 最关键的3个函数
1. **`OptLM.generate()`** (flex_opt.py:L825) - 生成入口
2. **`generation_loop_overlap_multi_batch()`** (flex_opt.py:L1034) - 高性能循环
3. **`TorchDevice.opt_attention()`** (pytorch_backend.py:L490) - 注意力计算

从这些入口和核心代码开始，就能完全理解FlexGen的工作原理！
