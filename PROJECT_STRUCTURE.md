# FlexLLMGen 项目结构说明

## 📁 项目目录结构

```
FlexLLMGen/
├── 📦 核心源代码
│   └── flexllmgen/              # 主包目录
│       ├── __init__.py
│       ├── compression.py       # 模型压缩实现
│       ├── dist_flex_opt.py     # 分布式FlexGen OPT实现
│       ├── dist_utils.py        # 分布式工具函数
│       ├── flex_opt.py          # FlexGen OPT模型核心
│       ├── opt_config.py        # OPT模型配置
│       ├── profile_bandwidth.py # 带宽性能分析
│       ├── profile_matmul.py    # 矩阵乘法性能分析
│       ├── pytorch_backend.py   # PyTorch后端实现
│       ├── timer.py             # 计时工具
│       ├── utils.py             # 通用工具函数
│       └── apps/                # 应用层
│           ├── __init__.py
│           ├── completion.py    # 文本生成补全
│           ├── helm_run.py      # HELM基准测试运行器
│           ├── helm_fast_test.py
│           ├── helm_passed_30b.sh
│           ├── README.md
│           └── data_wrangle/    # 数据整理应用
│
├── 🧪 基准测试
│   └── benchmark/
│       ├── batch_size_table.md
│       ├── flexgen/             # FlexGen基准测试
│       │   └── bench_scan_175b.sh
│       ├── flexllmgen/          # FlexLLMGen基准测试套件
│       │   ├── bench_suite.py   # 测试套件主程序
│       │   ├── bench_175b_1x4.sh    # OPT-175B (1节点4GPU)
│       │   ├── bench_175b_4x1.sh    # OPT-175B (4节点1GPU)
│       │   ├── bench_30b_1x4.sh     # OPT-30B (1节点4GPU)
│       │   ├── bench_30b_4x1.sh     # OPT-30B (4节点1GPU)
│       │   ├── bench_6.7b_1x4.sh    # OPT-6.7B (1节点4GPU)
│       │   ├── bench_6.7b_4x1.sh    # OPT-6.7B (4节点1GPU)
│       │   ├── bench_dist_multi_node.sh
│       │   ├── bench_dist_single_node.sh
│       │   └── README.md
│       ├── hf_ds/               # HuggingFace + DeepSpeed基准
│       │   ├── bench_hf.py
│       │   ├── hf_opt.py
│       │   ├── hostfile
│       │   ├── bench_all_1x4.sh
│       │   ├── bench_ds_175b_4x1.sh
│       │   ├── bench_ds_30b_1x4.sh
│       │   ├── bench_ds_30b_4x1.sh
│       │   ├── bench_ds_6.7b_1x4.sh
│       │   ├── bench_ds_6.7b_2x1.sh
│       │   ├── bench_ds_6.7b_4x1.sh
│       │   └── README.md
│       ├── petals/              # Petals基准测试
│       │   ├── run_opt_requests.py
│       │   └── README.md
│       └── third_party/         # 第三方依赖（本地副本）
│           ├── DeepSpeed/       # DeepSpeed框架
│           ├── transformers/    # HuggingFace Transformers
│           ├── pagecache-management/
│           └── README.md
│
├── 🔬 实验性代码
│   └── experimental/
│       ├── cost_model.py        # 成本模型
│       └── fit_cost_model.py    # 模型拟合
│
├── 📝 文档
│   └── docs/
│       ├── disk_commands.txt    # 磁盘操作命令
│       ├── gcp_setup.md        # GCP环境设置
│       └── paper.md            # 论文相关
│
├── 🛠️ 脚本工具
│   └── scripts/
│       ├── mount_nvme_aws.sh   # AWS NVMe挂载
│       ├── mount_nvme_gcp.sh   # GCP NVMe挂载
│       ├── step_2_consolidate_992_shards_to_singleton.py
│       ├── step_3_convert_to_numpy_weights.py
│       ├── upload_pypi.sh      # PyPI发布脚本
│       └── utils.py
│
└── 📄 配置文件
    ├── LICENSE                  # 许可证
    ├── README.md               # 项目说明
    └── pyproject.toml          # 项目配置
```

## 🎯 核心模块说明

### 1. **flexllmgen/** - 核心库
FlexLLMGen的主要实现代码，包括：
- **模型推理引擎**：`flex_opt.py`, `dist_flex_opt.py`
- **后端实现**：`pytorch_backend.py`
- **优化技术**：`compression.py`（模型压缩）
- **性能分析**：`profile_*.py`（性能测试）
- **应用层**：`apps/`（HELM、数据处理等应用）

### 2. **benchmark/** - 性能测试
包含多种基准测试套件：
- **flexllmgen/**: FlexLLMGen自己的基准测试
- **hf_ds/**: HuggingFace + DeepSpeed对比测试
- **petals/**: Petals框架对比测试
- **third_party/**: 第三方依赖库的本地副本

### 3. **experimental/** - 实验性功能
正在开发或测试中的功能，包括成本建模等

### 4. **scripts/** - 工具脚本
环境设置、模型转换、部署相关的辅助脚本

### 5. **docs/** - 文档
项目文档、设置指南、论文等

## 🚀 建议的优化方向

### 短期优化
1. **benchmark/third_party/** 可能过大，建议：
   - 使用git submodule管理第三方依赖
   - 或者使用requirements.txt直接从源安装

2. **添加缺失的文档**：
   - API文档（使用Sphinx生成）
   - 开发者指南
   - 贡献指南（CONTRIBUTING.md）

3. **测试代码组织**：
   - 创建 `tests/` 目录
   - 添加单元测试和集成测试

### 长期优化
1. **代码模块化**：
   - 将`flexllmgen/`拆分为子模块（core, backend, distributed, apps）
   
2. **CI/CD集成**：
   - 添加`.github/workflows/`用于自动化测试
   
3. **配置管理**：
   - 添加`configs/`目录统一管理配置文件

4. **示例代码**：
   - 创建`examples/`目录提供快速入门示例

## 📊 当前项目统计
- **主要语言**: Python
- **包版本**: 0.1.7
- **Python要求**: >=3.7
- **核心依赖**: PyTorch, Transformers, NumPy

## 🔗 相关资源
- 项目主页: https://github.com/FMInference/FlexLLMGen
- 论文: https://arxiv.org/abs/2303.06865
