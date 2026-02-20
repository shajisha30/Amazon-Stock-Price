import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="StockSense AI",
    page_icon="📊",
    layout="wide"
)

# ===============================
# PREMIUM FINTECH CSS
# ===============================
st.markdown("""
<style>

/* Import Professional Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Dark Gradient Background */
.stApp {
    background: linear-gradient(135deg, #0a0f1c, #05080f);
    color: white;
}

/* Subtle Chart Pattern Overlay */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: radial-gradient(circle at 1px 1px, rgba(0,255,255,0.05) 1px, transparent 0);
    background-size: 30px 30px;
    pointer-events: none;
}

/* Headings */
h1, h2, h3 {
    font-weight: 700;
    letter-spacing: 1px;
}

/* Glassmorphism Card */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(18px);
    border-radius: 16px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    transition: all 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 0 25px rgba(0,255,255,0.3);
}

/* Neon Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #00f5ff, #7b2ff7);
    border: none;
    color: white;
    font-weight: 600;
    border-radius: 12px;
    padding: 10px 20px;
    transition: 0.3s;
}

div.stButton > button:hover {
    box-shadow: 0 0 20px #00f5ff;
    transform: scale(1.05);
}

/* Chat Input Glass Style */
section[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-radius: 50px;
    border: 1px solid rgba(0,255,255,0.3);
    padding: 8px;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.03);
    border-radius: 15px;
    padding: 10px;
    margin-bottom: 10px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(#00f5ff, #7b2ff7);
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# HERO SECTION
# ===============================
st.markdown("# 🚀 StockSense AI")
st.markdown("### AI-Powered FinTech Stock Intelligence Platform")

# ===============================
# DASHBOARD CARDS
# ===============================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📈 Market Overview")
    st.metric("AMZN Price", "$182.34", "+1.45%")
    st.metric("Volume", "3.2M")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🤖 AI Confidence")
    st.progress(0.78)
    st.write("Prediction Confidence: **78%**")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📰 Sentiment Analysis")
    st.progress(0.65)
    st.write("News Sentiment: **Positive**")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ===============================
# AI CHATBOT SECTION
# ===============================
st.subheader("💬 Ask StockSense AI")

@st.cache_resource
def get_chain():
    model = OllamaLLM(model="gemma3:latest")

    template = """
    You are StockSense AI, a professional fintech assistant.
    Use ONLY provided stock records.
    Do not predict future prices.
    If data not available, say:
    "The dataset does not contain this information."

    Records:
    {records}

    Question:
    {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model

chain = get_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask about Amazon stock performance..."):

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing financial data..."):

            docs = retriever.invoke(question)
            records = "\n\n".join([doc.page_content for doc in docs])

            response = chain.invoke({
                "records": records,
                "question": question
            })

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.caption("StockSense AI • Next-Gen Financial Intelligence Dashboard")
