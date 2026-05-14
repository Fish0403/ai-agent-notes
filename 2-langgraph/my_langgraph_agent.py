"""
LangGraph 练习：一个可以执行 bash 命令的 Agent，支持人工确认后再执行。

这个文件是 1-mini-swe-agent/my_mini_agent.py 的 LangGraph 版本：
- my_mini_agent.py：手动 while 循环，query → execute
- my_langgraph_agent.py：用 StateGraph 替代手动循环，支持断点续传、人工确认

运行方式：
    python my_langgraph_agent.py

示例任务：
    - 列出当前目录文件
    - 查看 Python 版本
    - 创建一个测试文件
"""

import json
import operator
import os
import subprocess
from typing import Annotated
from typing_extensions import TypedDict
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

client = OpenAI(
    base_url="https://aihubmix.com/v1",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
)

MODEL = "coding-glm-5.1-free"

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


# ==================== 1. 定义 State ====================
class State(TypedDict):
    messages: Annotated[list, operator.add]
    step: int


# ==================== 2. 定义节点 ====================
def call_model(state: State) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=state["messages"],
        tools=[BASH_TOOL],
    )
    choice = response.choices[0]
    has_tools = choice.message.tool_calls
    print(f"[Step {state['step'] + 1}] tool_calls={bool(has_tools)}", end="")
    if not has_tools:
        print(f" content='{str(choice.message.content)[:80]}'")
    else:
        print()
    return {
        "messages": [choice.message.model_dump(exclude_unset=True)],
        "step": state["step"] + 1,
    }


def execute_tools(state: State) -> dict:
    last_message = state["messages"][-1]
    tool_results = []
    for tc in last_message.get("tool_calls", []):
        args = json.loads(tc["function"]["arguments"])
        cmd = args["command"]
        print(f"[Executing]: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout, end="")
        tool_results.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": json.dumps({
                "returncode": result.returncode,
                "output": result.stdout,
            }),
        })
    return {"messages": tool_results}


# ==================== 3. 条件路由 ====================
def route_after_model(state: State) -> str:
    last_message = state["messages"][-1]

    if state["step"] >= 3:
        return END

    if last_message.get("tool_calls"):
        return "execute_tools"

    content = last_message.get("content")
    if content:
        print(f"[Model said]: {content}")

    return END


# ==================== 4. 拼图 ====================
graph = StateGraph(State)

graph.add_node("call_model", call_model)
graph.add_node("execute_tools", execute_tools)

graph.add_edge(START, "call_model")
graph.add_conditional_edges("call_model", route_after_model)
graph.add_edge("execute_tools", "call_model")

app = graph.compile(
    checkpointer=InMemorySaver(),
    interrupt_before=["execute_tools"],  # 每次执行工具前暂停，等人工确认
)

# ==================== 5. 运行 ====================
task = input("Task: ").strip()
config = {"configurable": {"thread_id": "session-1"}}
state = {
    "messages": [
        {"role": "system", "content": "编程助手，可执行 bash 命令。"},
        {"role": "user", "content": task},
    ],
    "step": 0,
}

input_data = state   # 首次传初始 State，后续传 None
while True:
    result = app.invoke(input_data, config)   # 到达 execute_tools 前暂停
    input_data = None                         # 之后都是批准继续

    # 显示待执行的命令
    last_msg = result["messages"][-1]
    if last_msg.get("tool_calls"):
        pending = [json.loads(tc["function"]["arguments"])["command"] for tc in last_msg["tool_calls"]]
        print(f"\n--- 待执行: {' && '.join(pending)} ---")
    else:
        break   # 没有工具调用，模型完成了

    ans = input("执行？(y/n): ").strip().lower()
    if ans != "y":
        print("已取消")
        break
