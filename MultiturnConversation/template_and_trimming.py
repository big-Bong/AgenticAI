#Multi turn conversations with prompt template and maintaining limited conversation history

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState ,StateGraph

load_dotenv()

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a good therapist and like talking to people and making them comfortable. You try to understand \
            what a person is going through by talking with them."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)
trimmer = trim_messages(
    max_tokens=250,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)
messages = [
    SystemMessage(content="You are a good therapist and like talking to people and making them comfortable. You try to understand \
            what a person is going through by talking with them.")
]
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    print(f"Messages before trimming: {len(state['messages'])}")
    trimmed_messages = trimmer.invoke(state["messages"])
    print(f"Messages after trimming: {len(trimmed_messages)}")
    prompt = prompt_template.invoke(
        {"messages":trimmed_messages}
    )
    response = model.invoke(prompt)
    return {"messages": response}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

query = input("User: ")
while query != "bye":
    input_messages = messages + [HumanMessage(query)]
    output = app.invoke({"messages":input_messages}, config)
    output["messages"][-1].pretty_print()
    query = input("User: ")


