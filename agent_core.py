import json
from typing import List, Dict, Any, Callable, Optional
from nebius_client import NebiusNvidiaClient
from config import DEFAULT_MODEL

class AutonomousAgent:
    """
    Autonomous Agent powered by NVIDIA Nemotron-70B on Nebius Token Factory.
    Features tool calling, memory management, and structured execution.
    """
    def __init__(
        self,
        name: str = "NvidiaNebiusAgent",
        system_role: str = "You are an autonomous AI specialist running on NVIDIA Nemotron and Nebius infrastructure.",
        model: str = DEFAULT_MODEL
    ):
        self.name = name
        self.system_role = system_role
        self.model = model
        self.client = NebiusNvidiaClient()
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_role}
        ]
        self.tools: Dict[str, Callable] = {}
        self.tool_definitions: List[Dict[str, Any]] = []

    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], func: Callable):
        """Register a callable tool for the agent to use."""
        self.tools[name] = func
        self.tool_definitions.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        })

    def run(self, user_input: str, max_steps: int = 5) -> str:
        """
        Execute an agent reasoning step, optionally invoking tools.
        """
        self.history.append({"role": "user", "content": user_input})
        
        for step in range(max_steps):
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": self.history,
                "temperature": 0.5
            }
            if self.tool_definitions:
                kwargs["tools"] = self.tool_definitions
                kwargs["tool_choice"] = "auto"

            response = self.client.client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            
            # Check if tools were called
            if message.tool_calls:
                # Add assistant thought/tool call to history
                self.history.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                        } for tc in message.tool_calls
                    ]
                })
                
                # Execute each tool
                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments or "{}")
                    
                    if fn_name in self.tools:
                        try:
                            result = self.tools[fn_name](**args)
                        except Exception as err:
                            result = f"Error executing {fn_name}: {err}"
                    else:
                        result = f"Tool {fn_name} not found."
                        
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result
                    })
                continue
            else:
                final_content = message.content or ""
                self.history.append({"role": "assistant", "content": final_content})
                return final_content

        return "Max reasoning steps reached without final resolution."
