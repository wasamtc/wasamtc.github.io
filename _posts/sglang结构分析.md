---
layout:     post
title:      sglang结构分析
subtitle:   sglang的整体结构
date:       2025-11-26
author:     tangcong
header-img: img/sglang.jpg
catalog: true
tags:
    - sglang
---

# 1.从python -m sgalng.launch_server看整体运行

命令行运行python -m，运行python的一个模块而不是一个py文件

触发入口脚本`/python/sglang/launch_server.py`，代码如下

```python
"""Launch the inference server."""

import asyncio
import os
import sys

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree


def run_server(server_args):
    """Run the server based on server_args.grpc_mode."""
    if server_args.grpc_mode:
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    else:
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


if __name__ == "__main__":
    #  返回ServerArgs实例
    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)

```

调用launch_server函数，函数中会启动三个组件分别负责，分词，调度，反分词

```python
def launch_server(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """
    Launch SRT (SGLang Runtime) Server.

    The SRT server consists of an HTTP server and an SRT engine.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager all run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    tokenizer_manager, template_manager, scheduler_info, port_args = (
        _launch_subprocesses(server_args=server_args)
    )
```

在\_lauch\_subprocesses中启动三个组件，其中初始化调度器的时候会加载模型

`_launch_subprocesses` 函数是 SGLang Runtime (SRT) 引擎的核心启动函数，用于在主进程中初始化 TokenizerManager，并在子进程中启动 Scheduler 和 DetokenizerManager。它负责设置环境、分配通信端口、下载模型（如需要）、并等待模型加载完成。

\_launch\_subprocesses在`python/sglang/srt/entrypoints/engine.py`中，其流程如下

1. **配置环境和日志**：设置日志、验证参数、配置全局环境变量（如 CUDA、NCCL），支持并行通信。
2. **分配端口**：为进程间通信分配 IPC 端口（ZMQ），确保并行子进程间的通信。
3. **准备模型和 Tokenizer**：下载模型并更新路径，为并行加载做准备。
4. 启动调度器进程（并行处理）
   - 根据数据并行大小 (dp_size) 决定：
     - 如果 `dp_size == 1`：计算流水线并行 (PP) 和张量并行 (TP) rank 范围，为每个组合启动独立的 Scheduler 子进程（支持 TP、PP、EP 并行），分配 GPU ID 并行加载模型。
     - 如果 `dp_size > 1`：启动数据并行控制器进程，管理多个数据副本的并行推理。
5. **处理多节点情况（并行扩展）**：非主节点等待调度器就绪，支持多节点分布式并行（通过 `node_rank` 分配 rank）。
6. **启动 Detokenizer 进程**：创建反分词子进程（通常单进程，不涉及并行）。
7. **初始化 TokenizerManager**：根据配置初始化单/多 tokenizer 管理器，支持多 tokenizer 并行。
8. **等待模型加载完成**：从所有并行调度器管道读取状态，确保并行子进程模型加载完毕。
9. **返回结果**：返回管理器、信息和端口参数，完成并行引擎初始化。

# # 2.模型部署好后接受请求

当部署好模型并在某个端口监听时（例如通过 `launch_server` 启动），服务器由两个主要部分组成：**HTTP 服务器**（基于 FastAPI）和 **SRT 引擎**（包括 TokenizerManager、Scheduler 和 Detokenizer 子进程）。当一个请求到来时，处理流程如下：

### 2.1. **HTTP 服务器接收请求**

- **程序**：FastAPI HTTP 服务器（运行在主进程中）。
- **细节**：服务器在指定端口（例如 30000）监听 HTTP 请求。请求到达后，FastAPI 根据 URL 路径（如 `/v1/chat/completions`）路由到相应的 API 端点处理器（handlers）。
- **示例**：对于聊天完成请求，路由到 `openai_v1_chat_completions` 端点。

### 2.2. **API 端点处理器处理请求**

- **程序**：OpenAI 兼容的 serving handlers（例如 `OpenAIServingChat`、`OpenAIServingCompletion` 等）。
- **细节**：这些 handlers 验证请求格式（例如 JSON），然后调用 `TokenizerManager` 的 `generate_request` 方法，将请求转换为内部格式（如 `GenerateReqInput`）。
- **并行考虑**：如果配置了多 tokenizer（`tokenizer_worker_num > 1`），请求可能被路由到不同的 tokenizer worker 进程。

### 2.3. **TokenizerManager 分发请求**

- **程序**：TokenizerManager（运行在主进程中）。
- **细节**：TokenizerManager 对输入文本进行分词（tokenization），然后通过 ZMQ IPC 将请求发送到 Scheduler 子进程。Scheduler 是负责实际推理的核心组件。

### 2.4. **Scheduler 执行推理**

- **程序**：Scheduler 子进程（运行在子进程中，支持并行，如 TP、PP、DP）。
- 细节
  - Scheduler 接收请求，调度批次（batching），并在 GPU 上执行模型前向传播（inference）。
  - 如果是生成任务（generation），它会生成 token 序列；如果是嵌入任务（embedding），则计算向量。
  - 模型权重已在启动时加载到 Scheduler 子进程中。
  - 支持流式响应（streaming）：如果请求指定 `stream=True`，Scheduler 会逐步返回 token。

### 2.5. **Detokenizer 处理输出（如果需要）**

- **程序**：DetokenizerManager 子进程。
- **细节**：Scheduler 生成的 token 被发送到 Detokenizer，反分词为文本，然后通过 TokenizerManager 返回给客户端。

### 6. **结果返回给客户端**

- **程序**：HTTP 服务器。
- **细节**：最终结果通过 FastAPI 响应返回给客户端，支持 OpenAI 兼容的 JSON 格式或流式 SSE（Server-Sent Events）。

# 3.具体的处理请求逻辑

## OpenAIServingChat

OpenAIServingChat 是 SGLang 中处理 OpenAI 兼容的聊天完成请求（/v1/chat/completions）的核心类。它继承自 OpenAIServingBase，负责将 OpenAI 风格的请求转换为内部格式，并调用 SRT 引擎生成响应。

## 类结构和初始化

### 继承

继承自 OpenAIServingBase，共享通用逻辑（如请求验证、错误处理）。

### 初始化参数

- **tokenizer_manager**: TokenizerManager 实例，用于分词和推理调用。
- **template_manager**: TemplateManager 实例，用于处理聊天模板。

### 关键属性

- **tool_call_parser**: 工具调用解析器（从服务器参数获取）。
- **reasoning_parser**: 推理解析器（用于 DeepSeek 等模型的推理内容分离）。
- **default_sampling_params**: 从模型配置中获取的默认采样参数。

## 主要方法

### _validate_request
验证请求有效性，例如检查 messages 不为空，工具选择与工具列表一致等。  
返回错误消息字符串或 None（表示有效）。

### _convert_to_internal_request
将 ChatCompletionRequest 转换为内部 GenerateReqInput 格式。  
处理聊天消息、采样参数、工具调用、推理解析等。  
支持 LoRA 适配器、多模态输入（图像、音频、视频）。

（<font color="red">注意这里提到了LoRA</font>）

### _handle_streaming_request
处理流式请求（stream=True）。  
返回 StreamingResponse，逐步生成 token 并以 SSE 格式发送。  
处理 logprobs、工具调用、推理内容等。

### _handle_non_streaming_request
处理非流式请求。  
返回完整的 ChatCompletionResponse，包含所有生成的文本和元数据。

## 核心功能

### 兼容性
完全兼容 OpenAI API，支持消息格式、工具调用、函数调用、推理内容等。

### 多模态支持
处理图像、音频、视频输入。

### 工具集成
支持工具调用（tool calls），包括 JSON schema 约束。

### 推理解析
对于支持推理的模型（如 DeepSeek），分离推理内容和最终答案。

### 性能优化
使用异步处理，支持批处理和并行推理。

## 工作流程

1. 接收 ChatCompletionRequest。
2. 验证请求。
3. 转换为内部请求格式。
4. 调用 tokenizer_manager.generate_request() 执行推理。
5. 处理输出：格式化为 OpenAI 风格的响应，支持流式或一次性返回。

这个类是 HTTP 层与 SRT 引擎的桥梁，确保用户可以通过标准 OpenAI API 与 SGLang 交互。

# 4.ServingChat内部逻辑

内部使用_generate_chat_stream流式处理输入并返回输出

在这个函数内部再使用tokenizer_manager.generate_request进行推理

- **签名**：`async def generate_request(self, obj: Union[GenerateReqInput, EmbeddingReqInput], request: Optional[fastapi.Request] = None) -> AsyncGenerator[str, None]`
- **作用**：接收生成请求，执行分词、发送到调度器（Scheduler）进行推理，并异步返回结果。它支持流式输出（streaming），是连接HTTP层和推理引擎的关键桥梁。

### 主要流程

1. **初始化和预处理**：
   - 记录请求创建时间。
   - 调用`obj.normalize_batch_and_arguments()`标准化批处理参数。
   - 处理跟踪上下文（如果启用trace）：提取或设置远程传播上下文。
   - 如果启用多tokenizer worker，附加HTTP worker信息。
   - 记录请求日志（如果`log_requests`启用）。
2. **等待暂停状态**：
   - 使用`async with self.is_pause_cond:`等待，如果生成被暂停（`self.is_pause`为True），则阻塞直到继续。
3. **LoRA适配器处理**（如果启用）：
   - 如果请求包含`lora_path`，从`LoRARegistry`获取LoRA ID，并跟踪正在进行的LoRA请求<font color="red">注意这里有lora</font>>
4. **分派请求**：
   - **单个请求**：调用`_tokenize_one_request`进行分词，然后`_send_one_request`发送到调度器。
   - **批处理请求**：调用`_handle_batch_request`处理批次。
5. **等待和返回响应**：
   - 对于单个请求，调用`_wait_one_response`异步等待调度器的响应，并yield结果。
   - 对于批处理，类似处理多个请求。

### 关键子方法

- **`_tokenize_one_request`**：处理单个请求的分词，包括文本到token ID的转换、多模态输入处理、验证等。
- **`_send_one_request`**：通过ZMQ发送分词后的请求到Scheduler进程。
- **`_wait_one_response`**：异步等待响应，支持流式和非流式模式，处理断开连接、错误等。
- **`_handle_batch_request`**：处理批请求，包括批分词和并行发送。

### 异步生成器特性

- 返回`AsyncGenerator`，允许逐步yield结果，实现流式输出。
- 在`_wait_one_response`中，对于流式请求，会yield每个增量块；对于非流式，直接yield完整结果。

### 错误处理和清理

- 处理断开连接的请求（abort）。
- 跟踪请求状态（`rid_to_state`），确保资源清理。
- 支持LoRA请求的释放。

# 5.scheduler处理逻辑

在`event_loop_normal`中循环处理请求

`event_loop_normal` 是 SGLang SRT 服务器中 `Scheduler` 类的一个核心方法，用于实现正常的调度事件循环。它是一个无限循环，负责接收请求、处理输入、调度批次运行模型推理，并在空闲时进行自检。这个函数是调度器的主要执行逻辑，确保请求能够高效地通过 TokenizerManager 发送到 Scheduler 进行推理处理。

#### 函数签名和装饰器

- **定义位置**: `scheduler.py` 第 970 行左右（具体行号可能因版本而异）。
- **装饰器**: `@DynamicGradMode()` - 这是一个动态梯度模式装饰器，可能用于控制 PyTorch 的梯度计算模式，以优化推理性能。
- **返回类型**: 无返回值（`None`），因为这是一个无限循环的执行函数。

#### 函数概述

- 作用

  : 该函数实现了一个事件驱动的循环，用于处理来自 TokenizerManager 的请求。它是 Scheduler 的核心循环，负责：

  - 接收和处理新的请求。
  - 调度批次进行模型推理。
  - 处理推理结果并输出。
  - 在空闲时进行系统自检。

- **执行模式**: 这是“正常”模式的事件循环（相对应的是 `event_loop_overlap`，用于重叠调度以提高性能）。它不重叠 CPU 处理和 GPU 计算，而是顺序执行。

- 关键特性

  :

  - **无限循环**: 使用 `while True` 持续运行，直到进程终止。
  - **异步友好**: 虽然是同步循环，但与 asyncio 和 ZMQ 配合，支持高并发请求处理。
  - **并行支持**: 在多进程/多 GPU 环境中运行，支持 TP (Tensor Parallelism)、PP (Pipeline Parallelism) 等并行策略。

#### 代码步骤详细解释

函数体是一个简单的无限循环，每次迭代执行以下步骤：

1. **接收请求**:

   - 调用 `recv_requests()` 方法，从 ZMQ 套接字接收来自 TokenizerManager 的请求。
   - 这包括生成请求（`TokenizedGenerateReqInput`）、嵌入请求（`TokenizedEmbeddingReqInput`）或其他控制请求（如中止、刷新缓存等）。
   - 请求在 TP 组的 rank 0 上接收，然后广播到其他 TP ranks，确保所有并行进程同步。

2. **处理输入请求**:

   - 调用 `process_input_requests()` 方法，遍历接收到的请求。

   - 对于每个请求，使用

     ```
     _request_dispatcher
     ```

     （一个类型分发的调度器）根据请求类型调用相应的处理方法，例如：

     - `handle_generate_request()`: 处理生成请求，创建 `Req` 对象并添加到等待队列。
     - `handle_embedding_request()`: 处理嵌入请求。
     - 其他如 `abort_request()`、`flush_cache_wrapped()` 等控制请求。

   - 处理后，输出结果通过 ZMQ 发送回 TokenizerManager 或 DetokenizerManager。

3. **获取下一个要运行的批次**:

   - 调用 `get_next_batch_to_run()` 方法，从等待队列中选择下一个批次进行推理。
   - 这涉及调度策略（如优先级调度、连续批处理），考虑内存限制、KV 缓存等。
   - `batch` 是一个 `ScheduleBatch` 对象，包含要处理的请求列表。

4. **运行批次（如果有）**:

   - 如果有批次，调用

     ```
     run_batch()
     ```

     方法执行模型推理。

     - 这会调用 `TpModelWorker`（张量并行模型工作者）进行前向传播，生成 token 或嵌入。
     - 支持推测解码（speculative decoding）、混合内存等优化。

   - 然后调用

     ```
     process_batch_result()
     ```

     处理结果：

     - 更新请求状态、生成输出 token。
     - 通过 ZMQ 发送结果到 DetokenizerManager 进行输出处理。

   - 如果没有批次，跳到自检步骤。

5. **空闲时自检**:

   - 当没有批次运行时，调用 `self_check_during_idle()` 进行系统自检。
   - 这可能包括检查内存泄漏、重置状态、监控健康等，以确保系统稳定性。

6. **更新最后批次**:

   - 记录上一个批次，用于后续调度决策（如连续批处理优化）。

#### 涉及的关键方法和组件

- **`recv_requests()`**: 接收请求的核心方法，使用 ZMQ 非阻塞接收，并处理广播。
- **`process_input_requests()`**: 请求分发和处理，依赖 `_request_dispatcher`。
- **`get_next_batch_to_run()`**: 批次调度逻辑，考虑队列、优先级、内存等。
- **`run_batch()`**: 调用模型工作者执行推理，涉及 `TpModelWorker`。
- **`process_batch_result()`**: 结果后处理，更新请求和输出。
- **`self_check_during_idle()`**: 空闲自检，可能包括内存检查或状态重置。
- 相关类
  - `Req`: 请求对象，包含输入、采样参数等。
  - `ScheduleBatch`: 批次对象，包含多个请求。
  - `TpModelWorker`: 模型推理工作者，支持并行。
- **并行策略**: 函数在多进程环境中运行，支持 TP/PP/DP/EP ranks，通过 `self.tp_rank`、`self.pp_rank` 等属性协调。

#### 与 SRT 架构的关系

- **位置**: Scheduler 是 SRT 的推理调度层，位于 TokenizerManager（请求分发）和 DetokenizerManager（输出处理）之间。
- **数据流**: 请求从 HTTP 层 → TokenizerManager → Scheduler（此循环）→ DetokenizerManager → 客户端。
- **性能优化**: 这个循环是顺序的，但通过批处理和缓存（如 RadixCache）实现高效推理。相比 `event_loop_overlap`，它更简单，但可能在高负载下效率较低。
- **错误处理**: 如果推理失败，请求可能被中止并返回错误；循环本身不处理异常，依赖外部进程管理。

#### 示例执行流程

假设一个生成请求到达：

1. `recv_requests()` 接收请求。
2. `process_input_requests()` 创建 `Req` 并入队。
3. `get_next_batch_to_run()` 选择批次。
4. `run_batch()` 执行推理，生成 token。
5. `process_batch_result()` 发送结果。
6. 循环继续处理下一个请求。

run_batch内部调用model_worker.forward_batch_generation

调用model_worker.forward_batch_generation，内部调用model_runner.forward，

model_runner.forwar内部调用_forward_raw进行推理，\_forward_raw内部调用forward_decode，

forward_decode内部调用model.forward进行前向推理，从而进行模型推理的过程，之后感觉就跟vllm差不多了



