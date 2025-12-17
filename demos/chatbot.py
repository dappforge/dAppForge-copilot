import streamlit as st
from langchain_community.chat_models import BedrockChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up AWS credentials
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Initialize the BedrockChat model
model = BedrockChat(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.7,
    },
)

# Set page config
st.set_page_config(page_title="dApp AI Co-pilot", page_icon="🤖", layout="wide")

# Custom CSS to improve UI
st.markdown("""
<style>
.stTextInput > div > div > input {
    font-size: 16px;
}
.stTextArea > div > div > textarea {
    font-size: 16px;
}
.stButton > button {
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Streamlit app
st.title("🤖 dApp AI Co-pilot")
st.subheader("Your Substrate and ink! Coding Assistant")

# System message to guide the AI's behavior
system_message = """
You are dApp AI, an educational assistant designed to help users understand and use the dApp AI Co-pilot, which is a VSCode plugin for Substrate and ink! development. Your primary functions are:

1. Explain how to install and set up the dApp AI Co-pilot VSCode plugin, including the requirement for GitHub Token authentication.

2. Describe how the coding co-pilot works, specifically its code completion feature for "fill in the middle" scenarios where users provide prefix code in the coding IDE and the co-pilot completes the code when the user presses tab.

3. Provide information on best practices for using the co-pilot effectively with Substrate and ink! frameworks.

4. Answer questions about the capabilities and limitations of the dApp AI Co-pilot, focusing on its specialization in Substrate and ink! development.

5. Offer guidance on how users can leverage the co-pilot for learning and improving their Substrate and ink! coding skills.

When discussing the co-pilot's functionalities:
- Emphasize that it's designed for code completion, not for answering general coding questions.
- Explain that it works within the VSCode environment and requires proper setup and authentication.
- Highlight its specialization in Substrate and ink! frameworks.

Remember, you are not the actual code completion tool, but rather an educational assistant helping users understand and use the dApp AI Co-pilot effectively.
"""

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=system_message)
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages[1:]:  # Skip the system message
    if isinstance(message, HumanMessage):
        st.info(f"👤 You: {message.content}")
    else:
        st.success(f"🤖 dApp AI: {message.content}")

# Get user input
prompt = st.text_area("Enter your code or question:", key="user_input", height=150)

# React to user input
if st.button("Send", key="send_button"):
    if prompt:
        # Add user message to chat history
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)

        # Show user message
        st.info(f"👤 You: {prompt}")

        # Get chatbot response
        try:
            with st.spinner("dApp AI is thinking..."):
                response = model(st.session_state.messages)
            
            # Add assistant response to chat history
            assistant_message = AIMessage(content=response.content)
            st.session_state.messages.append(assistant_message)

            # Show AI response
            st.success(f"🤖 dApp AI: {response.content}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        # Clear the input area after sending by setting a flag
        st.session_state.clear_input = True
        st.experimental_rerun()

