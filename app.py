import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="College Football 2025 Rankings",
    page_icon="🏈",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("CFB_2025_CFP_Eligible_Final.csv")

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🏈 College Football 2025 Rankings")

st.write(
    "Custom college football ranking system based on record, "
    "strength of record, strength of schedule, quality wins, "
    "loss quality, scoring margin, Vegas performance, "
    "conference strength, location, and head-to-head results."
)

# --------------------------------------------------
# VIEW OPTIONS
# --------------------------------------------------

view = st.selectbox(
    "View",
    [
        "Top 25",
        "All CFP-Eligible Teams"
    ]
)

conference_options = (
    ["All Conferences"]
    + sorted(df["Conference"].dropna().unique().tolist())
)

selected_conference = st.selectbox(
    "Conference",
    conference_options
)

# --------------------------------------------------
# FILTER DATA
# --------------------------------------------------

if view == "Top 25":
    display_df = df.head(25).copy()
else:
    display_df = df.copy()

if selected_conference != "All Conferences":
    display_df = display_df[
        display_df["Conference"] == selected_conference
    ]

# --------------------------------------------------
# RANKINGS TABLE
# --------------------------------------------------

st.header(view)

table = display_df[
    [
        "Rank",
        "Team",
        "Conference",
        "Record",
        "Rating",
        "SOR Score",
        "SOS Score",
        "Win Resume",
        "Loss Quality"
    ]
].copy()

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# TEAM BREAKDOWN
# --------------------------------------------------

st.divider()

st.header("Team Breakdown")

selected_team = st.selectbox(
    "Select a team",
    df["Team"].tolist()
)

team_row = df[
    df["Team"] == selected_team
].iloc[0]

# --------------------------------------------------
# MAIN TEAM METRICS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Rank",
    f"#{int(team_row['Rank'])}"
)

col2.metric(
    "Record",
    team_row["Record"]
)

col3.metric(
    "Rating",
    f"{team_row['Rating']:.2f}"
)

col4.metric(
    "H2H Adjustment",
    f"{team_row['H2H Adjustment']:+.2f}"
)

# --------------------------------------------------
# RESUME METRICS
# --------------------------------------------------

st.subheader("Résumé")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Strength of Record",
    f"{team_row['SOR Score']:.1f}"
)

col2.metric(
    "Strength of Schedule",
    f"{team_row['SOS Score']:.1f}"
)

col3.metric(
    "Win Résumé",
    f"{team_row['Win Resume']:.1f}"
)

col4.metric(
    "Loss Quality",
    f"{team_row['Loss Quality']:.1f}"
)

# --------------------------------------------------
# PERFORMANCE METRICS
# --------------------------------------------------

st.subheader("Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Margin Score",
    f"{team_row['Margin Score']:.1f}"
)

col2.metric(
    "Vegas Score",
    f"{team_row['Vegas Score']:.1f}"
)

col3.metric(
    "Conference Score",
    f"{team_row['Conference Score']:.1f}"
)

col4.metric(
    "Location Win Score",
    f"{team_row['Location Win Score']:.1f}"
)

# --------------------------------------------------
# EXPLANATION
# --------------------------------------------------

st.subheader("Why They're Ranked Here")

st.write(
    team_row["Explanation"]
)

# --------------------------------------------------
# METHODOLOGY
# --------------------------------------------------

with st.expander("How the ranking works"):

    st.markdown(
        """
        The ranking system combines several résumé and performance factors.

        **Core résumé factors**
        - Strength of Record
        - Record
        - Strength of Schedule
        - Win Résumé

        **Context and performance**
        - Loss Quality
        - Scoring Margin
        - Performance vs. Vegas expectations
        - Location-adjusted wins
        - Conference strength

        **Head-to-head**
        - Head-to-head is used as a limited adjustment when similarly rated
          contenders played each other.
        - It does not override the full-season résumé.

        Postseason bowl and College Football Playoff games are excluded.
        """
    )