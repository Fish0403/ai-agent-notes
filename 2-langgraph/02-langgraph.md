# LangGraph

上一节用 LangChain 实现了简单的模型调用和工具绑定。但 chain 是线性的——从 A 到 B 到 C，一条路走到底。当 Agent 需要循环（调用工具后再问 LLM）、分支（根据结果走不同路径）、或者人工介入时，chain 就不够用了。

LangGraph 是 LangChain 团队的 Agent 编排框架。核心思想：把 Agent 的决策流程建模成**有向图**——节点负责"做什么"，边负责"做完之后去哪"。

一个简单的 Agent 循环（LLM 调用 → 工具执行 → LLM 再处理结果）本质是一个 while 循环。LangGraph 做的事情是把这条"线"变成"图"，支持条件分支、并行执行、断点续传等复杂编排。

下面介绍 LangGraph 的核心概念：State（共享状态）、Node（执行单元）、Edge（跳转关系）、StateGraph（图组装）、Checkpoint（断点续传）、Human-in-the-Loop（人工确认）。最后通过一个完整的 Agent 练手（`my_langgraph_agent.py`）。



## 1. State（状态）

State 是流经图中所有节点的共享数据。每个节点读取 State，返回 State 的**部分更新**（只返回自己修改的字段）。

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[list, operator.add]  # operator.add：new = old + new，追加而非覆盖
    step_count: int                           # 普通字段，新值覆盖旧值
```

**reducer**：当多个节点同时更新同一个 State 字段时，reducer 决定合并规则。

- `operator.add`：`list + list`，追加效果
- 不写 reducer 的字段：新值**覆盖**旧值
- 自定义 reducer：`Annotated[list, my_reducer]`，签名为 `(旧值, 新值) -> 合并后的值`

## 2. Node（节点）

Node 是图中的执行单元。每个 Node 是一个函数，签名为 `State → dict`。

```python
def call_model(state: State) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=state["messages"],
    )
    return {"messages": [response.choices[0].message]}
```

返回值只包含需要更新的字段，LangGraph 自动按 reducer 规则合并到全局 State。不需要手动维护消息列表。

## 3. Edge（边）

Edge 描述节点之间的跳转关系。有两种：

**普通边**：固定路线，源节点结束后必定去目标节点。

```python
graph.add_edge("call_model", END)         # call_model → 结束
graph.add_edge(START, "call_model")        # START → call_model（入口）
```

`START` 和 `END` 是 LangGraph 内置的哨兵常量，不是节点，不需要注册。`START` 只能作为边的起点，`END` 只能作为边的终点。

**条件边**：运行时根据函数返回值决定下一步去哪。

```python
def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    if last_message.get("tool_calls"):
        return "execute_tools"   # 有工具调用 → 去执行
    return END                   # 没有 → 结束

graph.add_conditional_edges("call_model", should_continue)
```

路径映射（第三个参数）的 key 是路由函数返回的字符串，value 是对应的目标节点名。如果路由函数返回值就能直接匹配节点名，可以省略此参数。

节点管"做什么"（State → dict），边管"去哪"（State → str）。两者职责明确分开。

## 4. StateGraph（图本身）

组装过程：

```python
from langgraph.graph import StateGraph, START, END

# 1. 建图
graph = StateGraph(State)

# 2. 注册节点
graph.add_node("call_model", call_model)
graph.add_node("execute_tools", execute_tools)

# 3. 连边
graph.add_edge(START, "call_model")
graph.add_conditional_edges("call_model", should_continue)
graph.add_edge("execute_tools", "call_model")   # 工具结果必须送回 LLM

# 4. 编译
app = graph.compile()

# 5. 运行
app.invoke({"messages": [{"role": "user", "content": "..."}]})
```

`compile()` 把节点和边转成 `Pregel` 运行时对象（参考 Google Pregel 图计算模型），会验证图完整性——死节点、不可达出口等错误在这一步暴露。

## 5. Checkpoint（断点续传）

每次节点执行后，LangGraph 自动保存当前 State 快照。中断后可从上次位置恢复，不需要重新开始。

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# InMemorySaver：进程退出后状态消失，适合测试
checkpointer = InMemorySaver()

# SqliteSaver：状态持久化到本地文件，重启后仍可恢复
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

app = graph.compile(checkpointer=checkpointer)

# config 中的 thread_id 是 checkpointer 定位状态的 key
# 同 ID → 同一条对话线；不同 ID → 独立会话
config = {"configurable": {"thread_id": "session-1"}}

# 首次 invoke，checkpointer 找不到 session-1，用输入初始化新 State
app.invoke({"messages": [{"role": "user", "content": "第一步"}]}, config)

# 同一 thread_id 再次 invoke，checkpointer 查到上次的 State，追加而非覆盖
app.invoke({"messages": [{"role": "user", "content": "继续"}]}, config)
# state["messages"] 包含两次 invoke 的全部历史

# 不带 config 时，每次 invoke 都是全新 session，checkpointer 无法跨 invoke 关联
app.invoke({"messages": [{"role": "user", "content": "这会是全新的"}]})
```

## 6. Human-in-the-Loop（人工确认）

关键操作需要人工确认时，用 `interrupt_before` 在指定节点前暂停：

```python
app = graph.compile(
    checkpointer=InMemorySaver(),          # interrupt 依赖 checkpointer
    interrupt_before=["execute_tools"],    # 每次执行工具前暂停
)

# 首次 invoke → 到达 execute_tools 前暂停
result = app.invoke(state, config)

# 显示待执行命令，人工决定
last_msg = result["messages"][-1]
if last_msg.get("tool_calls"):
    pending = [json.loads(tc["function"]["arguments"])["command"]
               for tc in last_msg["tool_calls"]]
    print(f"待执行: {' && '.join(pending)}")
    ans = input("执行？(y/n): ")

# 批准后，None = 无新输入，从断点继续
if ans == "y":
    app.invoke(None, config)
```

每次 `execute_tools` 被触发前都会暂停，即使同一次对话中有多轮工具调用。
