import os
import streamlit as st
from typing import Sequence
from typing_extensions import Annotated, TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# Define the supported models
SUPPORTED_MODELS = [
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "distil-whisper-large-v3-en",
    "llama-3.1-8b-instant",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "llama-3.2-90b-vision-preview",
    "llama-3.3-70b-specdec",
    "llama-3.3-70b-versatile",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
]

# Supported languages
SUPPORTED_LANGUAGES = ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Russian", "Arabic"]

# Streamlit app
def main():
    st.set_page_config(page_title="Linguo AI Assistant", page_icon="🤖")
    st.title("Linguo AI Assistant")
    initialize_session_state()
    configure_sidebar()
    for msg in st.session_state["messages"]:
        display_message(msg)
    user_input = st.chat_input("Type your message here...")
    if user_input:
        process_user_input(user_input)


def initialize_session_state():
    default_values = {
        "messages": [],
        "language": "English",
        "config": {"configurable": {"thread_id": "default_thread"}},
    }
    for key, value in default_values.items():
        st.session_state.setdefault(key, value)


def configure_sidebar():
    st.sidebar.header("Configuration")
    st.session_state["api_key"] = st.sidebar.text_input(
        "API Key:", type="password", placeholder="Enter your GROQ API key"
    )
    st.session_state["model_name"] = st.sidebar.selectbox(
        "Select Model:", SUPPORTED_MODELS, index=0
    )
    st.session_state["config"]["configurable"]["thread_id"] = st.sidebar.text_input(
        "Thread ID:", value=st.session_state["config"]["configurable"]["thread_id"]
    )
    st.session_state["language"] = st.sidebar.selectbox(
        "Language:", SUPPORTED_LANGUAGES, index=SUPPORTED_LANGUAGES.index(st.session_state["language"])
    )


def display_message(msg):
    if isinstance(msg, HumanMessage):
        sender = 'You'
        bg_color = '#bec7c7'
        align = 'right'
    elif isinstance(msg, AIMessage):
        sender = 'Assistant'
        bg_color = '#d6c372'
        align = 'left'
    else:
        return
    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            padding: 10px;
            border-radius: 15px;
            max-width: 70%;
            margin-bottom: 10px;
            float: {align};
            clear: both;
        ">
            <strong>{sender}:</strong> {msg.content}
        </div>
        """,
        unsafe_allow_html=True
    )


def process_user_input(user_input):
    human_message = HumanMessage(content=user_input)
    st.session_state["messages"].append(human_message)
    display_message(human_message)
    input_state = {
        "messages": st.session_state["messages"],
        "language": st.session_state["language"],
    }
    if "model" not in st.session_state:
        if not st.session_state.get("api_key"):
            st.error("Please enter your GROQ API key in the sidebar.")
            return
        st.session_state["model"] = ChatGroq(
            model=st.session_state["model_name"],
            api_key=st.session_state["api_key"]
        )
        initialize_workflow()
    with st.spinner("Assistant is typing..."):
        try:
            app = st.session_state["app"]
            output = app.invoke(input_state, st.session_state["config"])
            ai_message = output["messages"][-1]
            st.session_state["messages"].append(ai_message)
            display_message(ai_message)
        except Exception as e:
            st.error(f"An error occurred: {e}")


def initialize_workflow():
    general_prompt = ChatPromptTemplate.from_messages([
        ("system", """Hey there! 👋 Let’s keep this chill, respectful, and full of good vibes. I’m here to vibe with your style—whether you’re all about emojis, sarcasm, or straight-to-the-point clarity. I’ll keep it humble, sprinkle in some Gen Z/Millennial/Alpha slang (or dad jokes if you’re into that 😜), and adapt to how you communicate. Oh, and I’ll always serve responses in {language} to keep things smooth. Drop your thoughts, and let’s make this convo lit! 🚀 (P.S. No pressure—judgment-free zone here!)"

        Key Features:

        Tone: Friendly, approachable, and slightly playful (emojis, light humor, no formal jargon).

        Adaptability: Mirrors user’s typing style (e.g., short replies ↔️ paragraphs, slang ↔️ proper grammar).

        Inclusivity: Uses {language} naturally, respects cultural nuances, and avoids assumptions.

        Engagement: Asks curious follow-ups, shares relatable analogies, and celebrates user input.

        Example Interaction:

        User: "idk how to fix this lol"

        Response: "NGL, tech can be a mood 💀. Let’s troubleshoot—click the three dots and pray to the Wi-Fi gods? 🙏 Or describe the issue in {language}, and I’ll decode it!"

        User: "I need serious help with X."

        Response: "Got you. Let’s break this down step by step—no fluff. How does [specific detail] sound in {language}? 👀"""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        language: str
    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    def call_model(state: State):
        prompt = general_prompt.invoke({"messages": state["messages"], "language": state["language"]})
        response = st.session_state["model"].invoke(prompt)
        return {"messages": [response]}
    workflow.add_node("model", call_model)
    memory = MemorySaver()
    st.session_state["app"] = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    main()
