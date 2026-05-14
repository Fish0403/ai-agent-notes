"""
mini-swe-agent 精简复刻：只保留核心循环，对照学习用。

这是手动实现的 Agent 循环（while 循环 + query + execute）。
LangGraph 版本见 2-langgraph/my_langgraph_agent.py，用 StateGraph 替代手动循环，支持断点续传、人工确认。
"""

import json
import os
import subprocess
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ==================== 配置 ====================
client = OpenAI(
    base_url=os.getenv("AIHUBMIX_API_URL"),
    api_key=os.getenv("AIHUBMIX_API_KEY"),
)
MODEL = "coding-glm-5.1-free"
STEP_LIMIT = 10

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


# ==================== 异常 ====================
class Submitted(Exception):
    """环境检测到 COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT 信号，任务完成。"""
    pass


class FormatError(Exception):
    """模型输出格式不对（没调工具、工具名错、参数解析失败等）。"""
    def __init__(self, *messages: dict):
        super().__init__(str(messages))
        self.messages = list(messages)


# ==================== 核心函数 ====================
def query(messages: list[dict]) -> dict:
    """调用模型，解析 tool_calls，返回 assistant 消息。"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[BASH_TOOL],
    )
    choice = response.choices[0]
    tool_calls = choice.message.tool_calls or []

    actions = []
    for tc in tool_calls:
        args = json.loads(tc.function.arguments)
        actions.append({"command": args["command"], "tool_call_id": tc.id})

    if not actions:
        raise FormatError({
            "role": "user",
            "content": "你必须调用 bash 工具。如果任务完成，执行 echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT。",
        })

    message = {
        "role": "assistant",
        "content": choice.message.content or "",
        "extra": {"actions": actions},
    }
    messages.append(message)
    return message


def execute(messages: list[dict], message: dict):
    """逐个执行 action，结果追加到消息历史。"""
    for action in message["extra"]["actions"]:
        command = action["command"]
        tool_call_id = action["tool_call_id"]
        print(f"[Executing]: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout
        print(output, end="")

        lines = output.strip().splitlines()
        if lines and (lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" or lines[-1].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"):
            raise Submitted("Task complete")

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps({"returncode": result.returncode, "output": output}),
        })


def run(task: str):
    """主循环：query → execute → 检查退出条件。"""
    messages = [
        {"role": "system", "content": "你是一个编程助手，可以执行 bash 命令。完成后执行命令 echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT 来结束任务。"},
        {"role": "user", "content": task},
    ]

    for step in range(STEP_LIMIT):
        try:
            execute(messages, query(messages))
        except FormatError as e:
            print(f"[FormatError]: {e}")
            messages.extend(e.messages)
        except Submitted:
            print("Done.")
            return

    print("Out of steps.")


# ==================== 入口 ====================
if __name__ == "__main__":
    task = input("Task: ").strip()
    run(task)
