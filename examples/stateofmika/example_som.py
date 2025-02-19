from game_sdk.game.worker import Worker
from game_sdk.game.custom_types import FunctionResult
from game_sdk.plugins.stateofmika.functions.router import SOMRouterFunction


# Example state function
def get_state_fn(function_result: FunctionResult, current_state: dict) -> dict:
    """
    Update state based on router responses
    """
    init_state = {"previous_queries": [], "previous_routes": []}

    if current_state is None:
        return init_state

    # Update state with results
    if function_result and function_result.info:
        current_state["previous_queries"].append(function_result.info.get("query", ""))
        current_state["previous_routes"].append(function_result.info.get("route", {}))

    return current_state


# Initialize worker with SOM router
game_api_key = "your_game_api_key"

# Create router function
router_fn = SOMRouterFunction()

# Create worker
worker = Worker(
    api_key=game_api_key,
    description="An intelligent assistant that uses StateOfMika for routing queries",
    get_state_fn=get_state_fn,
    action_space=[router_fn],
)

# Run example query
worker.run("What's the latest price of Bitcoin?")
