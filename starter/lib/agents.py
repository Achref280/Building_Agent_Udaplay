import os
from typing import TypedDict, List, Optional, Union, TypeVar, Any
import json
from lib.vector_db import VectorStoreManager
from lib.state_machine import StateMachine, Step, EntryPoint, Termination, Run
from lib.llm import LLM
from lib.messages import AIMessage, UserMessage, SystemMessage, ToolMessage
from lib.tooling import Tool, ToolCall
from lib.memory import ShortTermMemory, LongTermMemory, MemoryFragment
from dotenv import load_dotenv

# Define the state schema
class AgentState(TypedDict):
    user_query: str  # The current user query being processed
    instructions: str  # System instructions for the agent
    messages: List[dict]  # List of conversation messages
    current_tool_calls: Optional[List[ToolCall]]  # Current pending tool calls
    total_tokens: int  # Track the cumulative total
class AgentStateGame(TypedDict):
    user_query: str # The current user query being processed
    documents: list[dict[str, Any]] | None# Retrieved document
    evaluation: dict[str, bool | str] | None# Retrieval quality
    web_result: dict[str, Any] | None# Web search results
    output: str | None 

class Agent:
    def __init__(self, 
                 model_name: str,
                 instructions: str, 
                 tools: List[Tool] = None,
                 temperature: float = 0.7):
        """
        Initialize an Agent
        
        Args:
            model_name: Name/identifier of the LLM model to use
            instructions: System instructions for the agent
            tools: Optional list of tools available to the agent
            temperature: Temperature parameter for LLM (default: 0.7)
        """
        self.instructions = instructions
        self.tools = tools if tools else []
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize memory and state machine
        self.memory = ShortTermMemory()
        self.workflow = self._create_state_machine()
        # Create long term memory
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        db = VectorStoreManager(
            os.getenv("OPENAI_API_KEY"),
            persist_path="./long_term_memory_db",
        )
        self.ltm = LongTermMemory(db)
        self.tools = [
            *self.tools,
            self.build_memory_registration_tool(),
            self.build_memory_search_tool(),
        ]
        

    def _prepare_messages_step(self, state: AgentState) -> AgentState:
        """Step logic: Prepare messages for LLM consumption"""
        messages = state.get("messages", [])
        
        # If no messages exist, start with system message
        if not messages:
            messages = [SystemMessage(content=state["instructions"])]
            
        # Add the new user message
        messages.append(UserMessage(content=state["user_query"]))
        
        return {
            "messages": messages,
            "session_id": state["session_id"]
        }

    def _llm_step(self, state: AgentState) -> AgentState:
        """Step logic: Process the current state through the LLM"""
        # Initialize LLM
        llm = LLM(
            model=self.model_name,
            temperature=self.temperature,
            tools=self.tools
        )

        response = llm.invoke(state["messages"])
        tool_calls = response.tool_calls if response.tool_calls else None

        current_total = state.get("total_tokens", 0)
        if response.token_usage:
            current_total += response.token_usage.total_tokens

        # Create AI message with content and tool calls
        ai_message = AIMessage(
            content=response.content, 
            tool_calls=tool_calls,
        )

        return {
            "messages": state["messages"] + [ai_message],
            "current_tool_calls": tool_calls,
            "session_id": state["session_id"],
            "total_tokens": current_total,
        }

    def _tool_step(self, state: AgentState) -> AgentState:
        """Step logic: Execute any pending tool calls"""
        tool_calls = state["current_tool_calls"] or []
        tool_messages = []
        
        for call in tool_calls:
            # Access tool call data correctly
            function_name = call.function.name
            function_args = json.loads(call.function.arguments)
            tool_call_id = call.id
            # Find the matching tool
            tool = next((t for t in self.tools if t.name == function_name), None)
            if tool:
                result = str(tool(**function_args))
                tool_message = ToolMessage(
                    content=json.dumps(result), 
                    tool_call_id=tool_call_id, 
                    name=function_name, 
                )
                tool_messages.append(tool_message)
        
        # Clear tool calls and add results to messages
        return {
            "messages": state["messages"] + tool_messages,
            "current_tool_calls": None,
            "session_id": state["session_id"]
        }

    def _create_state_machine(self) -> StateMachine[AgentState]:
        """Create the internal state machine for the agent"""
        machine = StateMachine[AgentState](AgentState)
        
        # Create steps
        entry = EntryPoint[AgentState]()
        message_prep = Step[AgentState]("message_prep", self._prepare_messages_step)
        llm_processor = Step[AgentState]("llm_processor", self._llm_step)
        tool_executor = Step[AgentState]("tool_executor", self._tool_step)
        termination = Termination[AgentState]()
        
        machine.add_steps([entry, message_prep, llm_processor, tool_executor, termination])
        
        # Add transitions
        machine.connect(entry, message_prep)
        machine.connect(message_prep, llm_processor)
        
        # Transition based on whether there are tool calls
        def check_tool_calls(state: AgentState) -> Union[Step[AgentState], str]:
            """Transition logic: Check if there are tool calls"""
            if state.get("current_tool_calls"):
                return tool_executor
            return termination
        
        machine.connect(llm_processor, [tool_executor, termination], check_tool_calls)
        machine.connect(tool_executor, llm_processor)  # Go back to llm after tool execution
        
        return machine

    def invoke(self, query: str, session_id: Optional[str] = None) -> Run:
        """
        Run the agent on a query
        
        Args:
            query: The user's query to process
            session_id: Optional session identifier (uses "default" if None)
            
        Returns:
            The final run object after processing
        """
        session_id = session_id or "default"
        # Create session if it doesn't exist
        self.memory.create_session(session_id)

        # Get previous messages from last run if available
        previous_messages = []
        last_run: Run = self.memory.get_last_object(session_id)
        if last_run:
            last_state = last_run.get_final_state()
            if last_state:
                previous_messages = last_state["messages"]

        initial_state: AgentState = {
            "user_query": query,
            "instructions": self.instructions,
            "messages": previous_messages,
            "current_tool_calls": None,
            "session_id": session_id,
        }

        run_object = self.workflow.run(initial_state)
        
        # Store the complete run object in memory
        self.memory.add(run_object, session_id)
        
        return run_object

    def get_session_runs(self, session_id: Optional[str] = None) -> List[Run]:
        """Get all Run objects for a session
        
        Args:
            session_id: Optional session ID (uses "default" if None)
            
        Returns:
            List of Run objects in the session
        """
        return self.memory.get_all_objects(session_id)

    def reset_session(self, session_id: Optional[str] = None):
        """Reset memory for a specific session
        
        Args:
            session_id: Optional session to reset (uses "default" if None)
        """
        self.memory.reset(session_id)
    
    def build_memory_registration_tool(self, owner:str="Achraf", namespace:str="Conversation"):
        """
        Create a tool for agents to register new memories.
        
        This factory function creates a tool that allows AI agents to store new
        information about users in the long-term memory system. The tool is
        pre-configured with specific owner and namespace parameters.
        
        Args:
            ltm (LongTermMemory): The memory system instance to use
            owner (str): User identifier for memory ownership
            namespace (str): Namespace for organizing memories
            
        Returns:
            Tool: A configured tool for memory registration
        """
        def _register(content:str):
            self.ltm.register(
                MemoryFragment(
                    content=content, 
                    owner=owner,
                    namespace=namespace
                )
            )
            return "Saved new memory"

        return Tool(
            func=_register, 
            name="register_memory", 
            description=(
                "Register a new memory or preference about the user, " 
                "so it can be useful later as context.\n"
                "Args:\n"
                "    content: The information to save"
            )
        )

    def build_memory_search_tool(self, owner:str="Achraf", namespace:str="Conversation"):
        """
        Create a tool for agents to search existing memories.
        
        This factory function creates a tool that allows AI agents to retrieve
        relevant memories from the long-term memory system based on semantic
        similarity to a search query.
        
        Args:
            ltm (LongTermMemory): The memory system instance to use
            owner (str): User identifier for memory ownership
            namespace (str): Namespace to search within
            
        Returns:
            Tool: A configured tool for memory search
        """
        def _search(content:str):
            result = self.ltm.search(
                query_text=content,
                owner=owner,
                namespace=namespace,
                limit=3,
            )
            return str(tuple(zip(result.fragments, result.metadata['distances'])))

        return Tool(
            func=_search, 
            name="search_memory", 
            description=(
                "Search for a stored memory or preference about the user, " 
                "so it's useful as a context.\n"
                "Args:\n"
                "    content: The information to look for"
            )
        )
class ToolNodeAgent(Agent):
    """
    An Agent whose state machine has a separate node for each tool.

    The original Agent uses one generic tool_executor node for every tool.
    This class inherits the original Agent's behavior but replaces its workflow
    with one that creates a named node for each tool.

    Example flow:
        message_prep
            -> llm_processor
            -> tool_search_memory
            -> llm_processor
            -> termination

    The LLM still decides which tool to call. Having separate nodes makes
    execution easier to inspect; it does not force a particular tool order.
    """

    def __init__(
        self,
        model_name: str,
        instructions: str,
        tools: Optional[List[Tool]] = None,
        temperature: float = 0.7,
    ):
        # Initialize everything provided by Agent:
        # - instructions and game tools
        # - short-term memory
        # - long-term memory and its tools
        # - the original workflow
        super().__init__(
            model_name=model_name,
            instructions=instructions,
            tools=tools,
            temperature=temperature,
        )

        # Replace this instance's original workflow with our new one.
        # At this point, the parent has already added the memory tools to
        # self.tools, so we can create nodes for those tools as well.
        self.workflow = self._create_tool_node_workflow()

    def _make_tool_step(self, tool: Tool):
        """
        Create the function that will run inside a node for one specific tool.

        For example, passing the search_memory tool produces a function that
        executes the next pending search_memory call.

        A separate function is created for each tool. The nested function
        remembers which 'tool' it was created for.
        """

        def execute(state: AgentState) -> dict:
            # The LLM can request one or more tool calls in a single response.
            # We process them one at a time because our state machine does
            # not support parallel execution.
            calls = state.get("current_tool_calls")

            if not calls:
                raise ValueError(f"No pending call for {tool.name}")

            # The first call is the one the router selected for this node.
            call = calls[0]

            # Catch a routing mistake instead of executing the wrong tool.
            if call.function.name != tool.name:
                raise ValueError(
                    f"Expected tool {tool.name}, got {call.function.name}"
                )

            # The model provides arguments as a JSON string.
            # Example: '{"content": "dark mode"}'
            arguments = json.loads(call.function.arguments)

            # Run this node's tool with those arguments.
            # Example: search_memory(content="dark mode")
            result = tool(**arguments)

            # Send the result back to the LLM as a tool message.
            # The ID connects this response to the original tool call.
            message = ToolMessage(
                content=json.dumps(str(result)),
                tool_call_id=call.id,
                name=tool.name,
            )

            # Return only the fields this node changes.
            # Step.run() merges these values into the existing state.
            return {
                "messages": state["messages"] + [message],

                # Remove the call we just handled. If no calls remain,
                # store None, which tells the router to return to the LLM.
                "current_tool_calls": calls[1:] or None,
            }

        return execute

    def _create_tool_node_workflow(self) -> StateMachine[AgentState]:
        """
        Build the state machine and connect its nodes.

        The route is:
            entry -> message_prep -> llm_processor

        After the LLM:
            - If there are tool calls, go to the requested tool node.
            - Otherwise, terminate.

        After a tool node:
            - If more tool calls remain, go to the next tool node.
            - Otherwise, return to the LLM with all tool results.
        """
        machine = StateMachine[AgentState](AgentState)

        # These are the shared nodes inherited from the original Agent.
        entry = EntryPoint[AgentState]()
        prepare = Step[AgentState](
            "message_prep",
            self._prepare_messages_step,
        )
        llm = Step[AgentState](
            "llm_processor",
            self._llm_step,
        )
        termination = Termination[AgentState]()

        # Create one node for each tool.
        #
        # Example:
        #   "search_memory" -> Step("tool_search_memory", ...)
        #
        # This dictionary allows the router to find a node by tool name.
        tool_nodes = {}

        for tool in self.tools:
            if tool.name in tool_nodes:
                raise ValueError(f"Duplicate tool name: {tool.name}")

            tool_nodes[tool.name] = Step[AgentState](
                f"tool_{tool.name}",
                self._make_tool_step(tool),
            )

        # Register every node with the state machine.
        machine.add_steps([
            entry,
            prepare,
            llm,
            *tool_nodes.values(),
            termination,
        ])

        # These first transitions always happen in the same order.
        machine.connect(entry, prepare)
        machine.connect(prepare, llm)

        def route_to_tool(state: AgentState) -> str:
            """
            Find the node corresponding to the first pending tool call.

            Return its step ID, for example "tool_search_memory".
            """
            calls = state.get("current_tool_calls")

            if not calls:
                raise ValueError("Expected a pending tool call")

            name = calls[0].function.name

            if name not in tool_nodes:
                raise ValueError(f"Unknown tool requested: {name}")

            return tool_nodes[name].step_id

        def after_llm(state: AgentState) -> str:
            """Choose between executing a tool and finishing the run."""
            if state.get("current_tool_calls"):
                return route_to_tool(state)

            return termination.step_id

        def after_tool(state: AgentState) -> str:
            """
            Execute the next pending tool, or return to the LLM.

            Returning to the LLM only after all pending calls are handled
            ensures each requested tool has a corresponding tool response.
            """
            if state.get("current_tool_calls"):
                return route_to_tool(state)

            return llm.step_id

        # The LLM can route to any tool node or finish.
        machine.connect(
            llm,
            [*tool_nodes.values(), termination],
            after_llm,
        )

        # Each tool can route to the next tool or back to the LLM.
        for node in tool_nodes.values():
            machine.connect(
                node,
                [*tool_nodes.values(), llm],
                after_tool,
            )

        return machine