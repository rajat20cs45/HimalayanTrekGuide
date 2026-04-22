import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.set_page_config(page_title="Himalayan Trek Guide", page_icon="🏔️", layout="wide")

# Background Image + Clean Styling
st.markdown("""
<style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1519681393784-d120267933ba');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .overlay {
        background: rgba(15, 23, 42, 0.92);
        padding: 40px;
        border-radius: 20px;
        margin: -40px -60px 40px -60px;
    }
    .main-header {
        font-size: 56px;
        background: linear-gradient(90deg, #bae6fd, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 700;
    }
    .sub-header {font-size: 25px; color: #e2e8f0; text-align: center; margin-bottom: 40px;}
    .card {
        background-color: rgba(30, 41, 59, 0.95);
        padding: 26px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        margin: 25px 0;
        color: white;
    }
    .difficulty {
        padding: 8px 22px;
        border-radius: 30px;
        font-weight: bold;
        display: inline-block;
        margin: 12px 0;
    }
    .bias-card {
        padding: 19px;
        border-radius: 12px;
        margin: 12px 0;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="overlay">', unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏔️ Himalayan Trek Guide</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Summarization • Responsible Bias-Aware Advice</p>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

tokenizer, model = load_model()

treks = {
    "Triund Trek (Himachal)": {"desc": "Triund is a beginner-friendly trek near Dharamshala offering panoramic views of the Dhauladhar range.", "difficulty": "Easy"},
    "Hampta Pass (Himachal)": {"desc": "Hampta Pass connects Kullu and Spiti with meadows, glaciers and high passes.", "difficulty": "Moderate"},
    "Valley of Flowers (Uttarakhand)": {"desc": "Known for its colorful alpine meadows and rare flowers.", "difficulty": "Moderate"},
    "Kedarkantha Trek (Uttarakhand)": {"desc": "Winter favorite with snow-covered peaks.", "difficulty": "Moderate"},
    "Markha Valley (Ladakh)": {"desc": "High altitude desert trek with Buddhist monasteries.", "difficulty": "Hard"}
}

st.sidebar.header("🌄 Choose Your Adventure")
selected_trek = st.sidebar.selectbox("Popular Himalayan Treks", list(treks.keys()))
data = treks[selected_trek]

input_text = st.text_area("Paste Trek Review or Description:", value=data["desc"], height=150)

if st.button("🔍 Analyze Trek", type="primary", use_container_width=True):
    if input_text.strip():
        with st.spinner("Analyzing trek..."):
            inputs = tokenizer("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
            summary_ids = model.generate(inputs.input_ids, max_length=160, min_length=80, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            diff_color = {"Easy": "#22c55e", "Moderate": "#eab308", "Hard": "#ef4444"}

            st.markdown(f"""
            <div class="card">
                <h3>{selected_trek}</h3>
                <span class="difficulty" style="background-color:{diff_color[data['difficulty']]};">{data['difficulty']} Difficulty</span>
                <p><strong>Clean AI Summary:</strong><br>{summary}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### ⚖️ Bias & Responsibility Audit")
            st.write("**How AI advice changes based on trekker identity:**")

            bias_data = [
                ("Solo Female Trekker", "Safety first. Trek with groups, share location, carry safety tools.", "#ef4444"),
                ("Beginner from Plains Trekker", "Acclimatize well. Altitude sickness risk. Hire guide.", "#f59e0b"),
                ("Experienced Local Trekker", "Check trail & weather. Pack light.", "#22c55e"),
                ("International Trekker", "Check permits & insurance. Budget for guides.", "#3b82f6"),
                ("Elderly Couple Trekker", "Choose easy routes. Take frequent breaks.", "#8b5cf6"),
                ("Family with Kids Trekker", "Focus on safety and fun. Avoid high altitude.", "#ec4899")
            ]

            cols = st.columns(3)
            for i, (title, advice, color) in enumerate(bias_data):
                with cols[i % 3]:
                    st.markdown(f"**{title}**")
                    st.markdown(f'<div class="bias-card" style="background-color:{color};">{advice}</div>', unsafe_allow_html=True)

            report_text = f"Trek: {selected_trek}\nDifficulty: {data['difficulty']}\n\nAI Summary:\n{summary}"
            st.download_button("📥 Download Report", report_text, file_name=f"{selected_trek}_report.txt", use_container_width=True)

    else:
        st.error("Please enter a trek description.")

st.markdown('</div>', unsafe_allow_html=True)
st.caption("Mini Project | T5 Summarization (Exp-1) + Bias Detection (Exp-6)")