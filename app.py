import streamlit as st
import pandas as pd
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="College Football Rankings",
    page_icon="🏈",
    layout="wide"
)

# --------------------------------------------------
# SEASON SELECTOR
# --------------------------------------------------

season = st.selectbox(
    "Season",
    [2026, 2025]
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

csv_file = f"CFB_{season}_CFP_Eligible_Final.csv"

if not os.path.exists(csv_file):
    st.warning(
        f"{season} rankings are not available yet. "
        "Showing 2025 rankings instead."
    )
    season = 2025
    csv_file = "CFB_2025_CFP_Eligible_Final.csv"

df = pd.read_csv(csv_file)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title(f"🏈 College Football {season} Rankings")

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
    width="stretch",
    hide_index=True
)


# --------------------------------------------------
# GAME PREDICTIONS
# --------------------------------------------------

st.divider()
st.header("🏈 Game Predictions")

predictions_file = "CFB_2026_Predictions.csv"

if not os.path.exists(predictions_file):
    st.info("2026 game predictions are not available yet.")
else:
    predictions = pd.read_csv(predictions_file)

    if predictions.empty:
        st.info("2026 game predictions are not available yet.")
    else:
        st.warning(
            "⚠️ Disclaimer: Weeks 1–4 predictions are based primarily on preseason "
            "and external rating models due to limited current-season data. "
            "The BEST predictions start Week 5, when our ranking system takes over. "
            "Be Ready. 🏈"
        )

        if "Official Pick" not in predictions.columns:
            predictions["Official Pick"] = False
        if "Result" not in predictions.columns:
            predictions["Result"] = ""

        official_flag = (
            predictions["Official Pick"]
            .fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )

        result_text = predictions["Result"].fillna("").astype(str).str.strip().str.lower()
        correct_mask = result_text.isin(
            ["correct", "win", "w", "true", "1", "✅ correct"]
        )
        incorrect_mask = result_text.isin(
            ["incorrect", "loss", "l", "false", "0", "❌ incorrect"]
        )
        graded_mask = correct_mask | incorrect_mask

        official_correct = int((official_flag & correct_mask).sum())
        official_incorrect = int((official_flag & incorrect_mask).sum())
        official_total = official_correct + official_incorrect
        official_accuracy = (
            official_correct / official_total * 100 if official_total else 0.0
        )

        early_flag = ~official_flag
        early_correct = int((early_flag & correct_mask).sum())
        early_incorrect = int((early_flag & incorrect_mask).sum())
        early_total = early_correct + early_incorrect
        early_accuracy = early_correct / early_total * 100 if early_total else 0.0

        st.subheader("Official Model Record — Week 5+")
        c1, c2, c3 = st.columns(3)
        c1.metric("Official Record", f"{official_correct}-{official_incorrect}")
        c2.metric("Official Accuracy", f"{official_accuracy:.1f}%")
        c3.metric("Official Games Graded", official_total)

        if early_total:
            st.caption(
                f"Early-season record (Weeks 1–4, not official): "
                f"{early_correct}-{early_incorrect} ({early_accuracy:.1f}%)."
            )
        else:
            st.caption(
                "Weeks 1–4 picks are tracked separately and do not count toward "
                "the official model record."
            )

        pending_mask = ~graded_mask
        upcoming = predictions[pending_mask].copy()

        def format_prediction_table(frame):
            preferred_cols = [
                "Week",
                "Away Team",
                "Home Team",
                "Model Pick",
                "Projected Home Margin",
                "Win Probability",
            ]
            cols = [c for c in preferred_cols if c in frame.columns]
            display = frame[cols].copy()

            if "Win Probability" in display.columns:
                probs = pd.to_numeric(display["Win Probability"], errors="coerce")
                if not probs.dropna().empty and probs.dropna().le(1).all():
                    probs = probs * 100
                display["Win Probability"] = probs.map(
                    lambda x: f"{x:.1f}%" if pd.notna(x) else ""
                )

            if "Projected Home Margin" in display.columns:
                margins = pd.to_numeric(
                    display["Projected Home Margin"], errors="coerce"
                )
                display["Projected Home Margin"] = margins.map(
                    lambda x: f"{x:+.1f}" if pd.notna(x) else ""
                )

            return display

        if not upcoming.empty:
            upcoming_official = official_flag.loc[upcoming.index]
            early_upcoming = upcoming[~upcoming_official].copy()
            official_upcoming = upcoming[upcoming_official].copy()

            if not early_upcoming.empty:
                st.subheader("Early-Season Predictions — Weeks 1–4")
                st.caption("These picks do not count toward the official model record.")
                st.dataframe(
                    format_prediction_table(early_upcoming),
                    width="stretch",
                    hide_index=True,
                )

            if not official_upcoming.empty:
                st.subheader("Official Upcoming Picks")
                st.dataframe(
                    format_prediction_table(official_upcoming),
                    width="stretch",
                    hide_index=True,
                )
        else:
            st.info("There are currently no ungraded predictions.")

        if graded_mask.any():
            st.subheader("Prediction History")

            history_cols = [
                "Week",
                "Away Team",
                "Home Team",
                "Model Pick",
                "Projected Home Margin",
                "Win Probability",
                "Official Pick",
                "Result",
            ]
            history_cols = [c for c in history_cols if c in predictions.columns]
            history = predictions.loc[graded_mask, history_cols].copy()

            if "Result" in history.columns:
                def format_result(value):
                    v = str(value).strip().lower()
                    if v in ["correct", "win", "w", "true", "1", "✅ correct"]:
                        return "✅ Correct"
                    if v in ["incorrect", "loss", "l", "false", "0", "❌ incorrect"]:
                        return "❌ Incorrect"
                    return str(value)

                history["Result"] = history["Result"].apply(format_result)

            if "Official Pick" in history.columns:
                history["Official Pick"] = (
                    history["Official Pick"]
                    .fillna(False)
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["true", "1", "yes"])
                    .map({True: "Yes", False: "No"})
                )

            if "Win Probability" in history.columns:
                probs = pd.to_numeric(history["Win Probability"], errors="coerce")
                if not probs.dropna().empty and probs.dropna().le(1).all():
                    probs = probs * 100
                history["Win Probability"] = probs.map(
                    lambda x: f"{x:.1f}%" if pd.notna(x) else ""
                )

            if "Projected Home Margin" in history.columns:
                margins = pd.to_numeric(
                    history["Projected Home Margin"], errors="coerce"
                )
                history["Projected Home Margin"] = margins.map(
                    lambda x: f"{x:+.1f}" if pd.notna(x) else ""
                )

            st.dataframe(history, width="stretch", hide_index=True)


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