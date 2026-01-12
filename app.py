import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import joblib
import numpy as np

# helper function to generate our field
def create_football_field(los_line_number=None, ydstogo=None, figsize=(12, 6.33)): # 5.33 + 1 for ease
    
    """
    - X-Axis (Length) -> 0 to 120 (0-10 is the left endzone, 
        10-110 is the main field, 
        110-120 is the right endzone)
    - Y-Axis (Width) -> 0 to 53.3 (width for an nfl field)
    """
    fig, ax = plt.subplots(1, figsize=figsize, facecolor='none')

    # main field background
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1, edgecolor='r', facecolor='darkgreen', zorder=0)
    ax.add_patch(rect)

    # draw the major yard lines in a snake like pattern
    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
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
        plt.text(x, 5, str(numb - 10), horizontalalignment='center', fontsize=20, color='white') # bottom numbers
        plt.text(x - 0.95, 53.3 - 5, str(numb - 10), horizontalalignment='center', fontsize=20, color='white', rotation=180) # top numbers upside down

    # hash marks
    for x in range(11, 110): # from the 1ydl to 110yd (opposite 1ydl)
        ax.plot([x, x], [0.4, 0.7], color='white') # bottom ydlines
        ax.plot([x, x], [53.0, 52.5], color='white') # top ydlines

        ax.plot([x, x], [22.91, 23.57], color='white') # hashmarks
        ax.plot([x, x], [29.73, 30.39], color='white')

    # plotting the LoS
    hl = los_line_number + 10 # hl = los line ,plus 10 for endzone
    plt.plot([hl, hl], [0, 53.3], color='yellow', linewidth=3)

    # first down marker
    if ydstogo:
        if hl > 100: # if we are inside the 10 then the first down line is the goal line (110)
            fd = 110 # fd = first down
            plt.plot([fd, fd], [0, 53.3], color='orange', linewidth=3, linestyle='--')
        else: # if we are not within goal
            fd = hl + ydstogo
            plt.plot([fd, fd], [0, 53.3], color='orange', linewidth=3, linestyle='--')
    
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')

    #fig.tight_layout(pad=0)
    return fig

# full browser width, set tab title
st.set_page_config(layout="wide", page_title="NFL Play Predictor")

st.markdown("""
<style>
    .metric-card {
        background-color: #222222;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .big-font { font-size: 24px !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource 
def load_model():
    return None

model = load_model()

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

    st.header("Game Situation Input 📋")

    quarter = st.select_slider("**Quarter**", options=[1, 2, 3, 4])
    minutes = st.number_input("**Minutes Remaining**", 0, 15, 10)
    score_diff = st.slider("**Score Differential**", -30, 30, 0)
    st.caption("*(ex. -7 is down by 7 pts.)*")

    st.markdown("---")

    st.header("📋 Drive Situation Input")
    down = st.selectbox("**Down**", [1, 2, 3, 4])
    distance = st.slider("**Yards to Go**", 1, 25, 10)
    yardline = st.slider("**Yardline** ", 1, 99, 50)
    st.caption("*(0 = Own Endzone, 100 = Opp Endzone)*")
    st.markdown("---")

    st.caption("Built by Ethan Wilson")

### main screen
st.title("🏈 ML Defensive Coordinator Assistant ")
st.markdown("Real-time play prediction based on historical NFL play-by-play data sourced from nflfastr.")

# input data for model
input_data = pd.DataFrame({
    'quarter': [quarter],
    'minutes_remaining': [minutes],
    'score_differential': [score_diff],
    'down': [down],
    'ydstogo': [distance],
    'yardline_100': [yardline]
})

### implement model here
if down == 4:
    # use the decision model
    pass
else:
    # use the prediction model
    pass 

# show the field
st.markdown("### 🏟️ Field View")
fig = create_football_field(los_line_number=yardline, ydstogo=distance)
st.pyplot(fig, transparent=True)