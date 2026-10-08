"""Restaurant order management agent built with LangGraph (uses Groq LLM)."""
from dotenv import load_dotenv

# Load GROQ_API_KEY from the .env file into environment variables (ChatGroq reads it automatically)
load_dotenv()

import random
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

MENU: dict[str, int] = {"burger": 10, "pizza": 20, "momos": 4}

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


class OrderExtraction(BaseModel):
    is_food_order: bool = Field(description="True only if the user wants to order food")
    dish_name: str = Field(default="", description="Single dish name, lowercase")
    required_quantity: int = Field(default=0, description="Quantity of that dish")


extractor = llm.with_structured_output(OrderExtraction)


class OrderState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    dish_name: str
    required_quantity: int
    available_quantity: int
    status: str  # pending, confirmed, partial, unavailable, ready, complete, failed
    order_retries: int
    cook_retries: int
    serve_retries: int
    final_result: bool


def say(text: str) -> AIMessage:
    print(f"\nAI: {text}")
    return AIMessage(content=text)


def llm_message(instruction: str, state: OrderState) -> AIMessage:
    """Ask the LLM to write a short customer-facing message."""
    reply = llm.invoke(
        [
            SystemMessage(content="You are a polite restaurant ordering assistant. Reply in 1-2 sentences."),
            HumanMessage(content=f"{instruction} (dish: {state['dish_name']}, qty: {state['required_quantity']})"),
        ]
    )
    return say(str(reply.content))


# ---------------------------------------------------------------- nodes
def llm_node(state: OrderState) -> dict:
    last_user = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    result: OrderExtraction = extractor.invoke(
        [
            SystemMessage(
                content=(
                    "You are an AI for food ordering ONLY. Decide whether the message is a food order. "
                    "If so, extract the single dish name (lowercase) and quantity. "
                    "Otherwise set is_food_order=false."
                )
            ),
            last_user,
        ]
    )
    if not result.is_food_order:
        msg = say("Sorry, I am an AI for food ordering only. I can't help with that.")
        return {"messages": [msg], "status": "failed", "final_result": False}
    if not result.dish_name or result.required_quantity <= 0:
        msg = say("I couldn't tell which dish and how many you want. Please try again.")
        return {"messages": [msg], "status": "failed", "final_result": False}
    return {
        "dish_name": result.dish_name.strip().lower(),
        "required_quantity": result.required_quantity,
        "status": "pending",
    }


def order_confirm(state: OrderState) -> dict:
    available = MENU.get(state["dish_name"], 0)
    if available == 0:
        status = "unavailable"
    elif state["required_quantity"] > available:
        status = "partial"
    else:
        status = "confirmed"
    return {"available_quantity": available, "status": status}


def user_decision(state: OrderState) -> dict:
    """Runs only for partial/unavailable orders: asks the user what to do."""
    retries = state["order_retries"] - 1
    if retries <= 0:
        msg = say("Sorry, we couldn't complete your order. We apologize for the inconvenience.")
        return {"messages": [msg], "order_retries": 0, "final_result": False, "status": "failed"}

    if state["status"] == "unavailable":
        print(f"\nAI: Sorry, '{state['dish_name']}' is unavailable.")
        choice = "new"
    else:
        print(
            f"\nAI: Only {state['available_quantity']} x {state['dish_name']} available "
            f"(you asked for {state['required_quantity']})."
        )
        choice = ""
        while choice not in ("partial", "new"):
            choice = input("Type 'partial' to proceed with available amount or 'new' for a new order: ").strip().lower()

    if choice == "partial":
        msg = say(f"Proceeding with {state['available_quantity']} x {state['dish_name']}.")
        return {
            "messages": [msg],
            "required_quantity": state["available_quantity"],
            "status": "confirmed",
            "order_retries": retries,
        }
    new_order = input("Enter your new order: ").strip()
    return {"messages": [HumanMessage(content=new_order)], "order_retries": retries}


def cook(state: OrderState) -> dict:
    if random.random() < 0.6:
        print("\n[kitchen] Cooking succeeded.")
        return {"status": "ready"}
    retries = state["cook_retries"] - 1
    print(f"\n[kitchen] Cooking failed. Retries left: {retries}")
    if retries <= 0:
        msg = llm_message("Apologize that the kitchen failed to cook the order", state)
        return {"messages": [msg], "cook_retries": 0, "status": "failed", "final_result": False}
    return {"cook_retries": retries, "status": "confirmed"}


def serve(state: OrderState) -> dict:
    if random.random() < 0.8:
        msg = llm_message("Tell the customer their order is served successfully", state)
        return {"messages": [msg], "status": "complete", "final_result": True}
    retries = state["serve_retries"] - 1
    print(f"\n[waiter] Serving failed, food ruined. Retries left: {retries}")
    if retries <= 0:
        msg = llm_message("Apologize that the order could not be served", state)
        return {"messages": [msg], "serve_retries": 0, "status": "failed", "final_result": False}
    if state["cook_retries"] <= 0:
        msg = llm_message("Apologize that the kitchen is out of attempts to cook again", state)
        return {"messages": [msg], "serve_retries": retries, "status": "failed", "final_result": False}
    return {"serve_retries": retries, "status": "confirmed"}


# ---------------------------------------------------------------- routing
def route_after_llm(state: OrderState) -> Literal["order_confirm", "__end__"]:
    return END if state["status"] == "failed" else "order_confirm"


def route_after_confirm(state: OrderState) -> Literal["cook", "user_decision"]:
    return "cook" if state["status"] == "confirmed" else "user_decision"


def route_after_decision(state: OrderState) -> Literal["cook", "llm_node", "__end__"]:
    if state["status"] == "failed":
        return END
    return "cook" if state["status"] == "confirmed" else "llm_node"


def route_after_cook(state: OrderState) -> Literal["serve", "cook", "__end__"]:
    if state["status"] == "ready":
        return "serve"
    return END if state["status"] == "failed" else "cook"


def route_after_serve(state: OrderState) -> Literal["cook", "__end__"]:
    return "cook" if state["status"] == "confirmed" else END


# ---------------------------------------------------------------- graph
builder = StateGraph(OrderState)
builder.add_node("llm_node", llm_node)
builder.add_node("order_confirm", order_confirm)
builder.add_node("user_decision", user_decision)
builder.add_node("cook", cook)
builder.add_node("serve", serve)

builder.set_entry_point("llm_node")
builder.add_conditional_edges("llm_node", route_after_llm, {"order_confirm": "order_confirm", END: END})
builder.add_conditional_edges("order_confirm", route_after_confirm, {"cook": "cook", "user_decision": "user_decision"})
builder.add_conditional_edges(
    "user_decision", route_after_decision, {"cook": "cook", "llm_node": "llm_node", END: END}
)
builder.add_conditional_edges("cook", route_after_cook, {"serve": "serve", "cook": "cook", END: END})
builder.add_conditional_edges("serve", route_after_serve, {"cook": "cook", END: END})

graph = builder.compile()


if __name__ == "__main__":
    while True:
        text = input("\nWhat would you like to order? (type 'quit' to exit): ").strip()
        if text.lower() in ("quit", "exit"):
            break
        initial: OrderState = {
            "messages": [HumanMessage(content=text)],
            "dish_name": "",
            "required_quantity": 0,
            "available_quantity": 0,
            "status": "pending",
            "order_retries": 3,
            "cook_retries": 2,
            "serve_retries": 2,
            "final_result": False,
        }
        final = graph.invoke(initial)
        print(f"\n=== Order result: {'SUCCESS' if final['final_result'] else 'FAILED'} (status={final['status']}) ===")
