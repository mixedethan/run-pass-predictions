import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import joblib
import lightgbm as lgb
import numpy as np
import shap

# helper function to generate our field
@st.cache_data(show_spinner=False)
def create_football_field(los_line_number=None, ydstogo=None):
    """
    - X-Axis (Length) -> 0 to 120 (0-10 is the left endzone, 
        10-110 is the main field, 
        110-120 is the right endzone)
    - Y-Axis (Width) -> 0 to 53.3 (width for an nfl field)
    """
    plt.style.use('fast')
    fig, ax = plt.subplots(1, figsize=(12,6), facecolor='none')

    # main field background
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1, edgecolor='r', facecolor='darkgreen', zorder=0)
    ax.add_patch(rect)

    # draw the major yard lines in a snake like pattern
    ax.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')

    # endzones
    ez1 = patches.Rectangle((0, 0), 10, 53.3, linewidth=0.1, edgecolor='r', facecolor='blue', alpha=0.2, zorder=0)
    ez2 = patches.Rectangle((110, 0), 120, 53.3,linewidth=0.1, edgecolor='r', facecolor='blue', alpha=0.2, zorder=0)
    ax.add_patch(ez1)
    ax.add_patch(ez2)

    # line numbers
    for x in range(20, 110, 10): # x coordinates
        numb = x
        if x > 50: numb = 120 - x # e.g. if x = 80, 120 - 80 = 40, opposite yardline
        ax.text(x, 5, str(numb - 10), horizontalalignment='center', fontsize=20, color='white') # bottom numbers
        ax.text(x - 0.95, 53.3 - 5, str(numb - 10), horizontalalignment='center', fontsize=20, color='white', rotation=180) # top numbers upside down

    # hash marks
    for x in range(11, 110): # from the 1ydl to 110yd (opposite 1ydl)
        ax.plot([x, x], [0.4, 0.7], color='white') # bottom ydlines
        ax.plot([x, x], [53.0, 52.5], color='white') # top ydlines

        ax.plot([x, x], [22.91, 23.57], color='white') # hashmarks
        ax.plot([x, x], [29.73, 30.39], color='white')

    # plotting the LoS
    hl = los_line_number + 10 # hl = los line ,plus 10 for endzone
    ax.plot([hl, hl], [0, 53.3], color='yellow', linewidth=3)

    # first down marker
    if ydstogo:
        if hl > 100: # if we are inside the 10 then the first down line is the goal line (110)
            fd = 110 # fd = first down
            ax.plot([fd, fd], [0, 53.3], color='orange', linewidth=3, linestyle='--')

        else: # if we are not within goal
            fd = hl + ydstogo
            ax.plot([fd, fd], [0, 53.3], color='orange', linewidth=3, linestyle='--')
    
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')

    #fig.tight_layout()
    return fig

@st.cache_resource 
def load_models():
    model_fp_decision = './models/decision_model.pkl'
    model_fp_play = './models/play_model.pkl'

    decision_model = joblib.load(model_fp_decision)
    play_model = joblib.load(model_fp_play)

    return decision_model, play_model

@st.cache_resource # to save some compute costs
def load_explainers(_decision_model, _play_model):
    decision_explainer = shap.TreeExplainer(_decision_model)
    play_explainer = shap.TreeExplainer(_play_model)
    return decision_explainer, play_explainer

def display_metric_box(col, title, value, sub_text=""):
    col.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub_text}</div>
        </div>
    """, unsafe_allow_html=True)

FEATURE_MAP = {
    "posteam_timeouts_remaining": "Timeouts",
    "yardline_100": "Field Position",
    "score_differential": "Score Diff",
    "game_seconds_remaining": "Time Left",
    "ydstogo": "Yards to Go",
    "down": "Down",
    "qtr": "Quarter"
}

# full browser width, set tab title
st.set_page_config(layout="wide", page_title="NFL Play Predictor")
# set page edges
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            margin-top: 1rem;
        }
        
        .css-1d391kg {
            padding-top: 1rem; 
        }
    </style>
""", unsafe_allow_html=True)
# define metric cards
st.markdown("""
<style>
    .metric-card {
        background-color: #222222;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        min-height: 140px;
        height: auto;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-title {
        color: #aaaaaa;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: clamp(18px, 2.5vw, 32px);
        font-weight: bold;
        color: #ffffff;
        margin: 5px 0;
        white-space: normal;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 12px;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

### --- sidebar -> game situation info ---
with st.sidebar:
    st.subheader("Connect with me")

    social_media_html = """
    <div style="display: flex; gap: 10px;">
        <a href="https://github.com/mixedethan" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" width="120" />
        </a>
        <a href="https://linkedin.com/in/ethan---wilson" target="_blank">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" width="120" />
        </a>
    </div>
    """
    st.markdown(social_media_html, unsafe_allow_html=True)
    
    st.markdown("---")

    with st.form(key='game_situation_form'):
        st.header("Game Situation Input")

        quarter = st.segmented_control("**Quarter**", options=[1, 2, 3, 4], selection_mode='single', default=1)
        minutes = st.number_input("**Minutes Remaining**", 0, 15, 15)

        # calculate game_seconds_remaining
        gsr = minutes * 60 + (4 - quarter) * 900
        score_diff = st.slider("**Score Differential**", -30, 30, 0)
        st.caption("*(ex. -7 is down by 7 pts.)*")
        to_remaining = st.segmented_control("**Timeouts Remaining**", options=[1, 2, 3], selection_mode='single', default=3)

        st.markdown("---")

        st.header("Drive Situation Input")
        down = st.segmented_control("**Down**", options=[1, 2, 3, 4], selection_mode='single')
        distance = st.slider("**Yards to Go**", 1, 25, 10)
        yardline = st.slider("**Yardline** ", 1, 99, 50)
        st.caption("*(0 = Own goal line, 100 = Opp goal line.)*")

        st.markdown("---")

        submit = st.form_submit_button(label='Predict Play💥')
        st.caption("*You must select a down to generate insights.*")
    
    st.markdown("---")
    st.caption("Engineered by Ethan Wilson")

### main screen
st.title("FourthDown.io 🏈")
st.markdown("""
<div style="margin-top: -20px; color: #aaaaaa; font-style: italic;">
    Leveraging gradient boosting and real-world NFL play-by-play data to predict offensive play-calling tendencies. Input a game situation to receive a defensive breakdown.
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# situational pills
tags = []

if yardline >= 80:
    tags.append(("🚨 Redzone Alert 🚨", "#ff4b4b"))
elif yardline <= 10:
    tags.append(("🧱 Backed Up", "#ffa500"))

if gsr <= 120 and score_diff < 0 and score_diff >= -8:
    tags.append(("⚡ 2-Minute Drill", "#00CC96"))

if down == 3 and distance >= 10:
    tags.append(("🛡️ Air Defense", "#1f77b4"))
elif down == 3 and distance <= 2:
    tags.append(("💪 Goal Line / Short Yardage", "#ff4b4b"))

if not tags:
    tags.append(("⚖️ Standard Play", "#333333"))

# display as pills
cols = st.columns(len(tags))
for col, (text, color) in zip(cols, tags):
    col.markdown(f"""
    <div style="
        background-color: {color};
        padding: 8px 15px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        text-align: center;
        font-size: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    ">
        {text}
    </div>
    """, unsafe_allow_html=True)
    

st.markdown("---")


#######
# input data for model
input_data = pd.DataFrame({
    'qtr': [quarter],
    'game_seconds_remaining': [gsr],
    'score_differential': [score_diff],
    'down': [down],
    'ydstogo': [distance],
    'yardline_100': [yardline],
    'posteam_timeouts_remaining': [to_remaining]
})

# we must convert the loaded dataframe into the exact same form as the trained models
input_data['down'] = input_data['down'].astype('category')
input_data['qtr'] = input_data['qtr'].astype('category')
cols_float = ['game_seconds_remaining', 'score_differential', 'ydstogo', 'yardline_100', 'posteam_timeouts_remaining']
for col in cols_float:
    input_data[col] = input_data[col].astype(float)

decision_model, play_model = load_models()
decision_explainer, play_explainer = load_explainers(decision_model, play_model)

# default values prior to user entering data
prediction_display = "N/A"
confidence_display = "0%"
key_factor_display = "Waiting..."

# nested models logic
if down is not None:
    if down == 4:
        # kick vs go
        features_decision = [
            'yardline_100', 'ydstogo', 'score_differential', 
            'game_seconds_remaining', 'posteam_timeouts_remaining'
        ]
            
        input_data_decision = input_data[features_decision].copy()
        cols_decision = ['yardline_100', 'ydstogo', 'score_differential', 'game_seconds_remaining', 'posteam_timeouts_remaining']
        input_data_decision = input_data_decision[cols_decision]

        prob_decision = decision_model.predict_proba(input_data_decision)[0] # probability e.g. [0.2, 0.8] 20% kick, 80% go
        pred_decision = decision_model.predict(input_data_decision)[0] # prediction e.g. 0 (kick) or 1 (go)

        # shap for decision (why go/kick?)
        shap_values = decision_explainer.shap_values(input_data_decision)

        if isinstance(shap_values, list): # if its a list
            vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            vals = shap_values[0]
        
        vals = np.atleast_2d(vals)

        idx = np.argmax(np.abs(vals[0]))
        decision_feature = input_data_decision.columns[idx]
        impact_decision = vals[0][idx]

        clean_name = FEATURE_MAP.get(decision_feature, decision_feature)

        if pred_decision == 0:
            # kick
            prediction_display = "KICK/PUNT"
            confidence_display = f"{prob_decision[0]*100:.1f}%" # kick probability
            key_factor_display = f"{clean_name} ({impact_decision:.2f})"

        else:
            # go for it
            cols_play = ['down', 'ydstogo', 'yardline_100', 'score_differential', 'game_seconds_remaining', 'posteam_timeouts_remaining', 'qtr']
            input_data_play = input_data[cols_play].copy()

            prob_play = play_model.predict_proba(input_data_play)[0]
            pred_play = play_model.predict(input_data_play)[0]
            
            play_call = "PASS" if pred_play == 1 else 'RUN'

            prediction_display = f"GO: {play_call}"
            confidence_display = f"Go: {prob_decision[1]*100:.0f}% | {play_call}: {max(prob_play)*100:.0f}%"   
            key_factor_display = f"{clean_name} ({impact_decision:.2f})"

    else: # down = [1, 2, 3]
        cols_play = ['down', 'ydstogo', 'yardline_100', 'score_differential', 'game_seconds_remaining', 'posteam_timeouts_remaining', 'qtr']
        input_data_play = input_data[cols_play].copy()

        prob_play = play_model.predict_proba(input_data_play)[0]
        pred_play = play_model.predict(input_data_play)[0]
        
        prediction_display = "PASS" if pred_play == 1 else "RUN"
        confidence_display = f"{max(prob_play)*100:.1f}%"
        
        shap_values = play_explainer.shap_values(input_data_play)
        
        if isinstance(shap_values, list):
            vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            vals = shap_values

        top_feature_name = input_data_play.columns[np.argmax(np.abs(vals[0]))]
        feature_impact = vals[0][np.argmax(np.abs(vals[0]))]
        
        # pretty print for UI
        direction = "⬆️" if feature_impact > 0 else "⬇️"
        clean_name = FEATURE_MAP.get(top_feature_name, top_feature_name)
        key_factor_display = f"{clean_name} {direction}"

col1, col2, col3 = st.columns(3)

# action prediction
display_metric_box(col1, "Predicted Call", prediction_display, "Based on tendencies")

# confidence
display_metric_box(col2, "Model Confidence", confidence_display, "Probability")

# why? / SHAP
display_metric_box(col3, "Key Factor", key_factor_display, "Biggest Influencer")


st.markdown("---")


# visuals for field and below
if down is not None:

    # generate the field
    st.markdown("### 🏟️ Field View")
    fig = create_football_field(los_line_number=yardline, ydstogo=distance)
    st.pyplot(fig, transparent=True, use_container_width=True)


    # probability distribution chart
    st.markdown("### 📊 Probability Breakdown")

    if down == 4 and pred_decision == 1: # if it's 4th and the model says go
        chart_data = pd.DataFrame({
            "Play Type": ["PASS", "RUN"],
            "Probability": [prob_play[1], prob_play[0]]
        })
        
        st.bar_chart(
            chart_data.set_index("Play Type"),
            color=["#FF4B4B"],
            horizontal=True
        )

    elif down == 4: # if it's 4th, give go vs kick probs
        # go vs kick probabilities
        chart_data = pd.DataFrame({
            "Decision": ["GO", "KICK/PUNT"],
            "Probability": [prob_decision[1], prob_decision[0]]
        })

        st.bar_chart(
            chart_data.set_index("Decision"), 
            color=["#00CC96"],
            horizontal=True
        )

    else: # 1-3 down, give pass/run probs
        chart_data = pd.DataFrame({
            "Play Type": ["PASS", "RUN"],
            "Probability": [prob_play[1], prob_play[0]]
        })

        st.bar_chart(
            chart_data.set_index("Play Type"),
            horizontal=True
        )

    # top 3 factors
    st.markdown("### 🧠 Model Reasoning (Top Factors)")

    # top 3 feature indices
    top_indices = np.argsort(np.abs(vals[0]))[-3:][::-1]

    # columns for the 3 factors
    f1, f2, f3 = st.columns(3)

    for i, col in zip(top_indices, [f1, f2, f3]):
        feature_name = input_data_decision.columns[i] if down == 4 else input_data_play.columns[i]
        feature_val = vals[0][i]
        clean_name = FEATURE_MAP.get(feature_name, feature_name)
        
        color = "green" if feature_val > 0 else "red"
        arrow = "⬆" if feature_val > 0 else "⬇"
        
        col.markdown(f"""
        <div style="background-color: #1E1E1E; padding: 10px; border-radius: 5px; border-left: 5px solid {color};">
            <small style="color: #aaa;">{clean_name}</small><br>
            <span style="font-size: 18px; font-weight: bold; color: white;">{arrow} Impact</span>
        </div>
        """, unsafe_allow_html=True)