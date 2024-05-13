import streamlit as st
import os
from DocumentReaderLangchain import DocGptLangResponse


from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI



openai_api_key = st.secrets.openai.api_key
# Side Bar Title
st.sidebar.title("Mini Project")

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.write("This features allows you chat with your custom uploaded 'Documents'")

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.markdown("## How to Use?")
st.sidebar.write("1. 'Upload your document'.")
st.sidebar.write("2. Ask any questions related to your document")
st.sidebar.write("3. Please ask your question in send section")


st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("## Project Member")
st.sidebar.write("1. Member-1")
st.sidebar.write("2. Member-2")
st.sidebar.write("3. Member-3")
st.sidebar.write("4. Member-4")

# Save uploaded file locally.
def save_uploaded_file(uploadedfile):
    with open(os.path.join("uploads", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())


def main():
    # Main Title
    st.title("Custom Video and YouTube Video Summarizer")

    st.markdown("<hr>", unsafe_allow_html=True)
    
    
    uploaded_file_doc_chat = st.file_uploader("Upload PDF file", type=["pdf"])
        
    if uploaded_file_doc_chat:
        
        # Set up memory
        msgs_dc = StreamlitChatMessageHistory(key="langchain_messages")
        if len(msgs_dc.messages) == 0:
            msgs_dc.add_ai_message("How can I help you! With this document?")
        # st.success('Your file successfully Submitted. S', icon="✅")
        st.markdown("<hr>", unsafe_allow_html=True)
        save_uploaded_file(uploaded_file_doc_chat)
        docgptlangresponse = DocGptLangResponse("uploads/{}".format(uploaded_file_doc_chat.name))
        doc_context = docgptlangresponse.doc_context()
        
        prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI chatbot having a conversation with a human regarding document context, Human can ask you question based on document context. following is the document context:"+doc_context),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
        )
        
        chain = prompt | ChatOpenAI(api_key=openai_api_key)
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: msgs_dc,
            input_messages_key="question",
            history_messages_key="history",
        )
        # Render current messages from StreamlitChatMessageHistory
        for msg in msgs_dc.messages:
            st.chat_message(msg.type).write(msg.content)
            
        # If user inputs a new prompt, generate and draw a new response
        if prompt := st.chat_input():
            st.chat_message("human").write(prompt)
            # Note: new messages are saved to history automatically by Langchain during run
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.invoke({"question": prompt}, config)
            st.chat_message("ai").write(response.content)

    else:
        st.warning("Please Upload the Docuemnt to chat.")



if __name__ == "__main__":
    main()