# Talk-To-Bigbook

针对超大文档（如《资治通鉴》）的非RAG检索方案，Vibe Coding 实现两种不同思路。

## 思路1：层次摘要压缩（layered-summary）

**核心思想**：书本多层次压缩精炼

### 索引结构
```
Level 3 (顶层): 1个全书摘要 (~1000 tokens)
    ↓
Level 2: 多个区域摘要 (~5000 tokens total)
    ↓
Level 1: 滑动窗口摘要 (window=5, stride=3)
    ↓
Level 0 (底层): 原始chunks (~2000 tokens each)
```

### 构建过程
1. 文档分chunk（2000 tokens，200 overlap）
2. Level 1：滑动窗口（5个chunk一组，步长3）生成摘要，保留跨chunk关联
3. Level 2：将Level 1摘要分组压缩
4. Level 3：生成全书摘要

### 查询过程
1. LLM看Level 3摘要，选择相关的Level 2节点
2. 展开选中的Level 2，选择相关的Level 1节点
3. 展开选中的Level 1，获取对应的原始chunks
4. 用chunks作为上下文生成答案

## 思路2：目录导航（category-navigate）

**核心思想**：按语义边界形成多层次主题目录树

### 索引结构
```
TOC (多层目录):
  Part (大章节): title + children
    ↓
  Section (小节): title + chunk_ids
    ↓
  Chunks: 原始文本片段
```

### 构建过程
1. 文档分chunk
2. **边界检测**：LLM判断相邻chunk是否讨论同一主题（SAME/DIFFERENT）
3. 根据边界将chunks分组为sections，每个section生成title（无摘要，纯目录风格）
4. **递归分层**：如果sections太多（>10），自动分组成更高层级的parts

### 查询过程
1. LLM看TOC顶层parts的标题，选择相关的
2. 展开选中的parts，看其子标题
3. 选择相关sections，获取对应chunks
4. 用chunks作为上下文生成答案

## 两者对比

| 维度 | 思路1 层次摘要 | 思路2 目录导航 |
|------|---------------|---------------|
| 分组依据 | 位置（滑动窗口） | 语义（主题边界） |
| 节点内容 | 纯摘要（内容压缩） | 纯标题（目录名） |
| 检索依据 | 摘要内容相关性 | 标题名称相关性 |
| 索引构建速度 | 较快（只需摘要） | 较慢（需边界检测） |
| 类比 | 视频压缩 | 文件夹目录 |

## 运行

### 安装依赖

```sh
pip install -r requirements.txt
```

如果使用本地 qwen2.5:7b 模型构建索引，先安装Ollama，然后:
```sh
ollama pull qwen2.5:7b
```

### 思路1 层次摘要压缩（layered-summary）

```bash
cd layered-summary

# 构建索引（使用DashScope在线API更快）
python examples/build_index.py ./data/documents/资治通鉴.txt \
    -o ./data/indexes/资治通鉴_index_1.json \
    --provider dashscope \
    --dashscope-key "xxx"

# 查询
python examples/query.py ./data/indexes/资治通鉴_index_1.json \
    "历史上有没有less is more的事情"
    --gemini-key xxx \
    --gemini-model gemini-3-flash-preview \
    --show-sources
```

### 思路2 目录导航（category-navigate）

```bash
cd category-navigate/examples

# 构建索引
python examples/build_index.py ./data/documents/资治通鉴.txt \
    -o ./data/indexes/资治通鉴_index_2.json \
    --provider dashscope \
    --dashscope-key "xxx"

# 查询
python examples/query.py ./data/indexes/资治通鉴_index_2.json \
    "历史上有没有less is more的事情"
    --gemini-key xxx \
    --gemini-model gemini-3-flash-preview \
    --show-sources
```
