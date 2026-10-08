I am building a restaurant order management AI agent system using LangGraph in Python. I will explain the architecture, nodes, state, and edges clearly. Write the complete, executable code based on these rules. Ask any clarifying questions at the end. (I will provide test cases to simulate later).

**1. The Goal and The LLM**
* There will be a user providing text input to place an order.
* Use `ChatOpenAI` (or your default preferred LLM) as the model. 
* The LLM's job is strictly food ordering. If the user input is unrelated to food, the LLM must reject it, stating it is an AI for food ordering only, and not process further.
* The user will order ONE particular dish and a specific quantity. The LLM must extract the `dish_name` and `required_quantity`.


**2. The Menu (Dictionary)**
Use the following hardcoded menu in your logic:
`MENU = {"burger": 10, "pizza": 20, "momos": 4}`
**3. LangGraph State Definition**
Create a TypedDict for the graph state containing:
* `messages`: Annotated list of messages between user and LLM.
* `dish_name`: str
* `required_quantity`: int
* `available_quantity`: int (Default to 0. If a dish is not on the menu, set this to 0).
* `status`: str (Tracks current progress: "pending", "confirmed", "partial", "unavailable", "ready", "complete", "failed").
* `order_retries`: int (Initialize to 3)
* `cook_retries`: int (Initialize to 2)
* `serve_retries`: int (Initialize to 2)
* `final_result`: bool (Whether the order was successfully completed or not).

**4. Nodes & Execution Logic**

* **LLM Node**: Extracts the dish and required quantity from the user's message. If successful, passes to `order_confirm`.
* **order_confirm Node**: Checks the extracted dish against the `MENU`. 
  * If fully available -> Update status to "confirmed".
  * If partially available (required > available) -> Update status to "partial".
  * If not available (not on menu or 0 stock) -> Update status to "unavailable".
* **User Decision Node (Conditional Edge)**:
  * If status is "confirmed", route to the `cook` node.
  * If status is "partial" or "unavailable", prompt the user to either place a new order or proceed with the partial amount.
  * Decrement `order_retries`. If `order_retries == 0`, apologize, set `final_result=False`, and route to `END`.
  * *Edge Case handling*: If the user proceeds with a partial order, update `required_quantity` to equal `available_quantity` before routing to `cook`.
* **cook Node**: 
  * Simulates cooking with a probability function: 60% chance of success, 40% chance of failure.
  * If success: Update status to "ready" and route to `serve` node.
  * If failure: Decrement `cook_retries`. 
    * If `cook_retries > 0`, retry the cook node.
    * If `cook_retries == 0`, issue an apology via the LLM, set `final_result=False`, and route to `END`.
* **serve Node**: 
  * Simulates serving with a probability function: 80% chance of success, 20% chance of failure.
  * If success: Update status to "complete", issue a success message to the user, set `final_result=True`, and route to `END`.
  * If failure: Decrement `serve_retries`.
    * If `serve_retries == 0`, issue an apology, set `final_result=False`, and route to `END`.
    * *Special Rule*: If serve fails but `serve_retries > 0`, the food is ruined. The system must route backward to the `cook` node to cook a new batch. This MUST check if `cook_retries > 0`. If the cook is out of retries, issue an apology, set `final_result=False`, and route to `END`.

**5. Output Requirements**
Please write the fully typed Python code for this LangGraph system. Ensure you include the graph compilation and a simple `while True` loop to take user input for testing. If any of the retry logic or cyclic edges seem contradictory to LangGraph's strict constraints, let me know in your open questions.