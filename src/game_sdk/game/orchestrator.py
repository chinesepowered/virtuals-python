# orchestrator.py

from typing import Dict, List, Optional
from collections import deque
import uuid
from pydantic import BaseModel
from game_sdk.game.worker import Worker
from game_sdk.game.custom_types import (
    Function, 
    FunctionResult, 
    FunctionResultStatus, 
    ActionResponse, 
    ActionType
)
from game_sdk.game.api import GAMEClient
from game_sdk.game.api_v2 import GAMEClientV2

class MessageContent(BaseModel):
    """Data model for inter-agent messages"""
    content: dict
    metadata: Optional[dict] = None

class SendMessageFunction(Function):
    """Function for sending messages between agents"""
    def __init__(self, orchestrator: 'AgentOrchestrator', agent_id: str):
        super().__init__()
        self.orchestrator = orchestrator
        self.from_agent = agent_id
        
    def get_function_def(self):
        return {
            "fn_name": "send_message",
            "description": "Send a message to another agent",
            "parameters": {
                "type": "object",
                "properties": {
                    "to_agent": {
                        "type": "string",
                        "description": "ID of the agent to send the message to"
                    },
                    "content": {
                        "type": "object",
                        "description": "Content of the message to send"
                    }
                },
                "required": ["to_agent", "content"]
            }
        }
        
    def execute(self, to_agent: str, content: dict) -> FunctionResult:
        try:
            message = MessageContent(content=content)
            self.orchestrator.send_message(self.from_agent, to_agent, message)
            return FunctionResult(
                action_id="send_message",
                action_status=FunctionResultStatus.DONE,
                feedback_message=f"Message sent to {to_agent}",
                info={"to": to_agent, "message": content}
            )
        except Exception as e:
            return FunctionResult(
                action_id="send_message",
                action_status=FunctionResultStatus.ERROR,
                feedback_message=str(e),
                info={"error": str(e)}
            )

class AgentSession:
    """Manages session state for an agent"""
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.function_result: Optional[FunctionResult] = None
        self.message_queue: deque = deque()
        self.is_active: bool = True
        self.waiting_for: Optional[str] = None

    def reset(self):
        self.id = str(uuid.uuid4())
        self.function_result = None
        self.message_queue.clear()
        self.is_active = True
        self.waiting_for = None

class AgentOrchestrator:
    """
    Orchestrates multiple agents' execution and interaction.
    Maintains agent states and handles message passing between agents.
    """
    def __init__(self):
        self.agents: Dict[str, 'MultiAgent'] = {}
        self.active_agents = deque()
        self.sessions: Dict[str, AgentSession] = {}

    def add_agent(self, agent_id: str, agent: 'MultiAgent'):
        """Add a new agent to the orchestrator"""
        self.agents[agent_id] = agent
        self.sessions[agent_id] = AgentSession()
        self.active_agents.append(agent_id)
        agent._orchestrator = self
        agent._id = agent_id

    def remove_agent(self, agent_id: str):
        """Remove an agent from the orchestrator"""
        if agent_id in self.active_agents:
            self.active_agents.remove(agent_id)
        if agent_id in self.agents:
            del self.agents[agent_id]
        if agent_id in self.sessions:
            del self.sessions[agent_id]

    def send_message(self, from_agent: str, to_agent: str, message: MessageContent):
        """Send a message from one agent to another"""
        if to_agent not in self.sessions:
            raise ValueError(f"Target agent {to_agent} not found")
        
        session = self.sessions[to_agent]
        session.message_queue.append({
            "from": from_agent,
            "content": message.content,
            "metadata": message.metadata
        })

    def _get_next_active_agent(self) -> Optional[str]:
        """Get the next agent to execute in round-robin fashion"""
        if not self.active_agents:
            return None
            
        next_agent = self.active_agents.popleft()
        session = self.sessions[next_agent]
        
        if session.waiting_for is not None:
            self.active_agents.append(next_agent)
            return None
            
        return next_agent

    def step(self) -> bool:
        """Execute one step of the multi-agent system"""
        next_agent_id = self._get_next_active_agent()
        if next_agent_id is None:
            return len(self.active_agents) > 0

        current_agent = self.agents[next_agent_id]
        session = self.sessions[next_agent_id]

        # Process pending messages
        while session.message_queue:
            message = session.message_queue.popleft()
            current_agent.agent_state["messages"] = current_agent.agent_state.get("messages", [])
            current_agent.agent_state["messages"].append(message)

        # Get and execute action
        action_response = current_agent._get_action(session.function_result)
        
        if action_response.action_type in [ActionType.CALL_FUNCTION, ActionType.CONTINUE_FUNCTION]:
            if not action_response.action_args:
                raise ValueError("No function information provided by GAME")

            function_result = (
                current_agent.workers[current_agent.current_worker_id]
                .action_space[action_response.action_args["fn_name"]]
                .execute(**action_response.action_args)
            )

            session.function_result = function_result
            current_agent._session.function_result = function_result

            # Update worker state
            current_worker = current_agent.workers[current_agent.current_worker_id]
            updated_worker_state = current_worker.get_state_fn(
                function_result,
                current_agent.worker_states[current_agent.current_worker_id]
            )
            current_agent.worker_states[current_agent.current_worker_id] = updated_worker_state

        elif action_response.action_type == ActionType.WAIT:
            session.is_active = False
            return len(self.active_agents) > 0

        elif action_response.action_type == ActionType.GO_TO:
            if "location_id" in action_response.action_args:
                current_agent.current_worker_id = action_response.action_args["location_id"]

        if session.is_active:
            self.active_agents.append(next_agent_id)

        current_agent.agent_state = current_agent.get_agent_state_fn(
            session.function_result,
            current_agent.agent_state
        )

        return True

    def run(self):
        """Run the multi-agent system until all agents are done"""
        while self.step():
            pass

class MultiAgent:
    """
    Enhanced version of the Agent class with multi-agent capabilities.
    Includes support for message passing and interaction with other agents.
    """
    def __init__(
        self,
        api_key: str,
        name: str,
        agent_goal: str,
        agent_description: str,
        get_agent_state_fn: callable,
        workers: Optional[List[WorkerConfig]] = None,
    ):
        if api_key.startswith("apt-"):
            self.client = GAMEClientV2(api_key)
        else:
            self.client = GAMEClient(api_key)

        self._api_key = api_key
        if not self._api_key:
            raise ValueError("API key not set")

        self._session = AgentSession()
        self._orchestrator: Optional[AgentOrchestrator] = None
        self._id: Optional[str] = None

        self.name = name
        self.agent_goal = agent_goal
        self.agent_description = agent_description

        if workers is not None:
            self.workers = {w.id: w for w in workers}
        else:
            self.workers = {}
        self.current_worker_id = None

        self.get_agent_state_fn = get_agent_state_fn
        self.agent_state = self.get_agent_state_fn(None, None)

        self.agent_id = self.client.create_agent(
            self.name, self.agent_description, self.agent_goal
        )

    def compile(self):
        """Compile the workers and add messaging capability"""
        if not self.workers:
            raise ValueError("No workers added to the agent")

        # Add message sending capability to each worker
        for worker_config in self.workers.values():
            if self._orchestrator and self._id:  # Only add if agent is part of an orchestrator
                message_fn = SendMessageFunction(self._orchestrator, self._id)
                worker_config.action_space.append(message_fn)

        workers_list = list(self.workers.values())
        self._map_id = self.client.create_workers(workers_list)
        self.current_worker_id = next(iter(self.workers.values())).id

        # Initialize worker states
        worker_states = {}
        for worker in workers_list:
            dummy_function_result = FunctionResult(
                action_id="",
                action_status=FunctionResultStatus.DONE,
                feedback_message="",
                info={},
            )
            worker_states[worker.id] = worker.get_state_fn(
                dummy_function_result, self.agent_state)

        self.worker_states = worker_states
        return self._map_id

    def _get_action(self, function_result: Optional[FunctionResult] = None) -> ActionResponse:
        """Get next action from GAME API"""
        if function_result is None:
            function_result = FunctionResult(
                action_id="",
                action_status=FunctionResultStatus.DONE,
                feedback_message="",
                info={},
            )

        data = {
            "location": self.current_worker_id,
            "map_id": self._map_id,
            "environment": self.worker_states[self.current_worker_id],
            "functions": [
                f.get_function_def()
                for f in self.workers[self.current_worker_id].action_space
            ],
            "events": {},
            "agent_state": self.agent_state,
            "current_action": (
                function_result.model_dump(exclude={'info'})
                if function_result else None
            ),
            "version": "v2",
        }

        response = self.client.get_agent_action(
            agent_id=self.agent_id,
            data=data,
        )

        return ActionResponse.model_validate(response)

    def add_worker(self, worker_config: WorkerConfig):
        """Add a worker to the agent"""
        self.workers[worker_config.id] = worker_config
        return self.workers

    def get_worker_config(self, worker_id: str):
        """Get worker configuration"""
        return self.workers[worker_id]

    def get_worker(self, worker_id: str):
        """Get a standalone worker instance"""
        worker_config = self.get_worker_config(worker_id)
        return Worker(
            api_key=self._api_key,
            description=self.agent_description,
            instruction=worker_config.instruction,
            get_state_fn=worker_config.get_state_fn,
            action_space=worker_config.action_space,
        )

# Usage example:
if __name__ == "__main__":
    def get_state_fn(result, current):
        return {"state": "example"}
        
    # Create orchestrator
    orchestrator = AgentOrchestrator()
    
    # Create first agent
    agent1 = MultiAgent(
        api_key="your-api-key",
        name="Agent1",
        agent_goal="Example goal 1",
        agent_description="Example description 1",
        get_agent_state_fn=get_state_fn
    )
    
    # Create second agent
    agent2 = MultiAgent(
        api_key="your-api-key",
        name="Agent2",
        agent_goal="Example goal 2",
        agent_description="Example description 2",
        get_agent_state_fn=get_state_fn
    )
    
    # Add workers to agents
    worker1 = WorkerConfig(
        id="worker1",
        worker_description="Worker 1",
        get_state_fn=get_state_fn,
        action_space=[],  # Add your custom functions here
        instruction="Example instruction 1"
    )
    
    worker2 = WorkerConfig(
        id="worker2",
        worker_description="Worker 2",
        get_state_fn=get_state_fn,
        action_space=[],  # Add your custom functions here
        instruction="Example instruction 2"
    )
    
    agent1.add_worker(worker1)
    agent2.add_worker(worker2)
    
    # Add agents to orchestrator
    orchestrator.add_agent("agent1", agent1)
    orchestrator.add_agent("agent2", agent2)
    
    # Compile agents
    agent1.compile()
    agent2.compile()
    
    # Run the multi-agent system
    orchestrator.run()